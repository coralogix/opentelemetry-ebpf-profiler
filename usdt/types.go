// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package usdt discovers User Statically-Defined Tracepoint probes in ELF
// mappings.
package usdt // import "go.opentelemetry.io/ebpf-profiler/usdt"

// AttachmentPoint describes a discovered USDT location. Location and
// SemaphoreOffset are ELF file offsets suitable for uprobe attachment.
type AttachmentPoint struct {
	Provider        string
	Name            string
	Location        uint64
	SemaphoreOffset uint64
	// Arguments is the raw SDT argument descriptor, e.g. "8@%rdi -4@-8(%rbp)".
	Arguments string
}
