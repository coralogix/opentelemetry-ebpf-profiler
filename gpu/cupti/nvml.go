// SPDX-License-Identifier: Apache-2.0

package cupti // import "go.opentelemetry.io/ebpf-profiler/gpu/cupti"

import (
	"context"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"go.opentelemetry.io/ebpf-profiler/internal/log"
)

// BusyPoller samples per-process GPU utilization via `nvidia-smi pmon`
// (util% × interval = estimated GPU busy time). Covers CUDA processes
// without the shim injected. nvidia-smi instead of libnvidia-ml because the
// profiler builds CGO_ENABLED=0.
type BusyPoller struct {
	report   func(pid uint32, busyNs int64)
	interval time.Duration
	argv     []string
}

// NewBusyPoller locates nvidia-smi (PATH, then the host rootfs for
// containerized agents with hostPID); nil if unavailable.
func NewBusyPoller(report func(pid uint32, busyNs int64)) *BusyPoller {
	if p, err := exec.LookPath("nvidia-smi"); err == nil {
		return &BusyPoller{report: report, interval: 5 * time.Second,
			argv: []string{p, "pmon", "-c", "1"}}
	}
	for _, hostPath := range []string{"/usr/bin/nvidia-smi", "/usr/local/nvidia/bin/nvidia-smi"} {
		if _, err := os.Stat("/proc/1/root" + hostPath); err == nil {
			// Host binary needs the host's libnvidia-ml → enter its mount ns,
			// running the discovered host path (it may not be in PATH there).
			if ns, err := exec.LookPath("nsenter"); err == nil {
				return &BusyPoller{report: report, interval: 5 * time.Second,
					argv: []string{ns, "-t", "1", "-m", "--", hostPath, "pmon", "-c", "1"}}
			}
		}
	}
	return nil
}

func (b *BusyPoller) poll() {
	// nvidia-smi hangs when the driver is wedged; never block the poller.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, b.argv[0], b.argv[1:]...).Output()
	if err != nil {
		log.Debugf("gpu/cupti: nvidia-smi pmon: %v", err)
		return
	}
	// pmon output: "# gpu pid type sm mem enc dec ... command"; '-' = idle.
	for _, line := range strings.Split(string(out), "\n") {
		if line == "" || line[0] == '#' {
			continue
		}
		f := strings.Fields(line)
		if len(f) < 4 {
			continue
		}
		pid, err := strconv.ParseUint(f[1], 10, 32)
		if err != nil || pid == 0 {
			continue
		}
		sm, err := strconv.Atoi(f[3])
		if err != nil || sm <= 0 { // '-' or 0
			continue
		}
		b.report(uint32(pid), int64(sm)*b.interval.Nanoseconds()/100)
	}
}
