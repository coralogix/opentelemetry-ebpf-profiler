// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu

import (
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cilium/ebpf/link"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/usdt"
	"go.opentelemetry.io/ebpf-profiler/util"
)

func TestCheckArguments(t *testing.T) {
	pinned := map[string]string{"amd64": "8@%rax", "arm64": "8@x0"}[runtime.GOARCH]
	tests := map[string]struct {
		arguments string
		valid     bool
	}{
		"pinned register": {arguments: pinned, valid: true},
		"other register":  {arguments: "8@%rdi"},
		"arm64 other reg": {arguments: "8@x1"},
		"memory operand":  {arguments: "8@64(%rcx)"},
		"narrow register": {arguments: "4@%eax"},
		"two arguments":   {arguments: pinned + " " + pinned},
		"no arguments":    {arguments: ""},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			err := checkArguments(usdt.AttachmentPoint{
				Provider:  usdtProvider,
				Name:      usdtProbe,
				Arguments: tc.arguments,
			})
			if tc.valid {
				assert.NoError(t, err)
			} else {
				assert.Error(t, err)
			}
		})
	}
}

// fakeLink is a uprobe link that counts its closes.
type fakeLink struct {
	link.Link
	closed *atomic.Int32
}

func (l fakeLink) Close() error {
	l.closed.Add(1)
	return nil
}

func newAttachTestProbe(t *testing.T) *probe {
	t.Helper()
	p, err := New(Config{})
	require.NoError(t, err)
	gp := p.(*probe)
	gp.matcher = newTestMatcher(t, liveProcesses{})
	return gp
}

// addFile registers uprobes for a file, as Attach does once they are attached.
func addFile(p *probe, file util.OnDiskFileIdentifier, closed *atomic.Int32) *fileProbes {
	probes := &fileProbes{links: []link.Link{fakeLink{closed: closed}, fakeLink{closed: closed}}}
	p.files[file] = probes
	return probes
}

func TestFileProbesShared(t *testing.T) {
	p := newAttachTestProbe(t)
	file := util.OnDiskFileIdentifier{DeviceID: 1, InodeNum: 2}
	var closed atomic.Int32
	probes := addFile(p, file, &closed)

	p.mu.Lock()
	p.addUser(10, file)
	p.addUser(11, file)
	p.addUser(11, file)
	p.mu.Unlock()
	assert.Equal(t, 2, probes.users)

	p.Detach(10)
	assert.Equal(t, 1, probes.users)
	assert.Nil(t, probes.idle, "still used")

	p.Detach(11)
	require.NotNil(t, probes.idle, "removal scheduled")
	assert.Empty(t, p.users)
	assert.Zero(t, closed.Load(), "kept for an application that restarts")

	// A process maps the file before the timeout: the uprobes stay.
	p.mu.Lock()
	p.addUser(12, file)
	p.mu.Unlock()
	p.removeUnused(file, probes)
	assert.Same(t, probes, p.files[file])

	p.Detach(12)
	p.removeUnused(file, probes)
	p.detaching.Wait()
	assert.NotContains(t, p.files, file)
	assert.Equal(t, int32(2), closed.Load())
}

func TestFileProbesRemovalCompletesBeforeReattach(t *testing.T) {
	p := newAttachTestProbe(t)
	file := util.OnDiskFileIdentifier{DeviceID: 1, InodeNum: 2}
	var closed atomic.Int32
	probes := addFile(p, file, &closed)

	p.mu.Lock()
	p.removeLocked(file, probes)
	p.waitRemovedLocked(file)
	p.mu.Unlock()
	assert.Equal(t, int32(2), closed.Load(), "old uprobes are gone before new ones")
	assert.NotContains(t, p.removing, file)
}

func TestFileProbesUnload(t *testing.T) {
	p := newAttachTestProbe(t)
	file := util.OnDiskFileIdentifier{DeviceID: 1, InodeNum: 2}
	var closed atomic.Int32
	addFile(p, file, &closed)
	p.mu.Lock()
	p.addUser(10, file)
	p.mu.Unlock()
	p.Detach(10)

	require.NoError(t, p.Unload())
	assert.Equal(t, int32(2), closed.Load())
	assert.Empty(t, p.files)
	// The idle timer firing after Unload changes nothing.
	p.removeUnused(file, &fileProbes{})
	time.Sleep(10 * time.Millisecond)
	assert.Equal(t, int32(2), closed.Load())
}

func TestAddUserRetiresReusedPID(t *testing.T) {
	p := newAttachTestProbe(t)
	now := time.Now()
	p.matcher.addLaunch(10, 1, hostTrace(0x10), now)
	p.matcher.processExited(10, now)

	file := util.OnDiskFileIdentifier{DeviceID: 1, InodeNum: 2}
	var closed atomic.Int32
	addFile(p, file, &closed)
	p.mu.Lock()
	p.addUser(libpf.PID(10), file)
	p.mu.Unlock()
	assert.NotContains(t, p.matcher.exits, libpf.PID(10), "the exited process is retired at once")
}

func TestCheckPointsAllowsOneSite(t *testing.T) {
	pinned := map[string]string{"amd64": "8@%rax", "arm64": "8@x0"}[runtime.GOARCH]
	site := usdt.AttachmentPoint{Provider: usdtProvider, Name: usdtProbe, Arguments: pinned}
	require.NoError(t, checkPoints([]usdt.AttachmentPoint{site}))
	require.Error(t, checkPoints([]usdt.AttachmentPoint{site, site}),
		"a library firing the probe from two places")
}

func TestPinnedFileProbesStay(t *testing.T) {
	p := newAttachTestProbe(t)
	file := util.OnDiskFileIdentifier{DeviceID: 1, InodeNum: 2}
	var closed atomic.Int32
	probes := addFile(p, file, &closed)
	probes.pinned = true

	p.mu.Lock()
	p.addUser(10, file)
	p.mu.Unlock()
	p.Detach(10)
	assert.Nil(t, probes.idle, "no removal scheduled for a configured producer")
	p.removeUnused(file, probes)
	assert.Same(t, probes, p.files[file])
	assert.Zero(t, closed.Load())
}
