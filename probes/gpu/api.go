// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu // import "go.opentelemetry.io/ebpf-profiler/probes/gpu"

import (
	"fmt"
	"maps"
	"path/filepath"
	"strings"

	cebpf "github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"

	"go.opentelemetry.io/ebpf-profiler/process"
	"go.opentelemetry.io/ebpf-profiler/tracer"
)

const (
	apiStartsMap    = "gpu_api_starts"
	apiEnterProgram = "uprobe__gpu_api_enter"
	apiExitProgram  = "uretprobe__gpu_api_exit"
)

// apiFunction is a function of a host GPU library whose calls become spans.
type apiFunction struct {
	// library is the file name prefix of the library exporting the function.
	library string
	symbol  string
	display string
}

// apiFunctions are the instrumented cuDNN and cuBLAS functions. The index is the
// attach cookie the eBPF programs report; functions missing from a library
// version are skipped.
var apiFunctions = []apiFunction{
	{"libcudnn.so", "cudnnConvolutionForward", "cudnn:ConvolutionForward"},
	{"libcudnn.so", "cudnnConvolutionBackwardData", "cudnn:ConvolutionBackwardData"},
	{"libcudnn.so", "cudnnConvolutionBackwardFilter", "cudnn:ConvolutionBackwardFilter"},
	{"libcudnn.so", "cudnnBatchNormalizationForwardTrainingEx", "cudnn:BatchNormForwardTrainingEx"},
	{"libcudnn.so", "cudnnBatchNormalizationForwardTraining", "cudnn:BatchNormForwardTraining"},
	{"libcudnn.so", "cudnnBatchNormalizationForwardInference", "cudnn:BatchNormForwardInference"},
	{"libcudnn.so", "cudnnBatchNormalizationBackwardEx", "cudnn:BatchNormBackwardEx"},
	{"libcudnn.so", "cudnnBatchNormalizationBackward", "cudnn:BatchNormBackward"},
	{"libcudnn.so", "cudnnPoolingForward", "cudnn:PoolingForward"},
	{"libcudnn.so", "cudnnPoolingBackward", "cudnn:PoolingBackward"},
	{"libcudnn.so", "cudnnActivationForward", "cudnn:ActivationForward"},
	{"libcudnn.so", "cudnnSoftmaxForward", "cudnn:SoftmaxForward"},
	{"libcudnn.so", "cudnnAddTensor", "cudnn:AddTensor"},
	{"libcudnn.so", "cudnnBackendExecute", "cudnn:BackendExecute"},
	{"libcublas.so", "cublasSgemm_v2", "cublas:Sgemm"},
	{"libcublas.so", "cublasGemmEx", "cublas:GemmEx"},
	{"libcublas.so", "cublasGemmStridedBatchedEx", "cublas:GemmStridedBatchedEx"},
	{"libcublas.so", "cublasSgemmStridedBatched", "cublas:SgemmStridedBatched"},
	{"libcublasLt.so", "cublasLtMatmul", "cublasLt:Matmul"},
}

// apiFunctionName returns the display name of the function with this cookie.
func apiFunctionName(cookie uint64) string {
	if cookie < uint64(len(apiFunctions)) {
		return apiFunctions[cookie].display
	}
	return "api:unknown"
}

// apiLibrary returns the library prefix of a mapping exporting instrumented
// functions, or "" if API spans are not loaded or the mapping is no such library.
func (p *probe) apiLibrary(mapping *process.RawMapping) string {
	if p.apiStarts == nil {
		return ""
	}
	name := filepath.Base(mapping.Path)
	for _, library := range []string{"libcudnn.so", "libcublasLt.so", "libcublas.so"} {
		if strings.HasPrefix(name, library) {
			return library
		}
	}
	return ""
}

// loadAPISpans loads the API span programs. They need BPF attach cookies, which
// arrived in Linux 5.15.
func (p *probe) loadAPISpans(probeCtx *tracer.ProbeContext) error {
	coll, err := probeCtx.CollectionSpecWith(
		[]string{eventsMap, apiStartsMap},
		[]string{apiEnterProgram, apiExitProgram},
		nil,
	)
	if err != nil {
		return err
	}
	starts, err := cebpf.NewMap(coll.Maps[apiStartsMap])
	if err != nil {
		return fmt.Errorf("creating %s: %w", apiStartsMap, err)
	}
	if err := probeCtx.RewriteMaps(coll, map[string]*cebpf.Map{
		eventsMap:    p.events,
		apiStartsMap: starts,
	}); err != nil {
		_ = starts.Close()
		return err
	}
	programs := make(map[string]*cebpf.Program)
	if err := probeCtx.LoadProbeUnwinders(coll, programs, []tracer.ProgLoaderHelper{
		{Name: apiEnterProgram, NoTailCallTarget: true, Enable: true},
		{Name: apiExitProgram, NoTailCallTarget: true, Enable: true},
	}, 0); err != nil {
		_ = starts.Close()
		return err
	}
	p.apiStarts = starts
	maps.Copy(p.programs, programs)
	return nil
}

// attachAPI attaches span probes to the instrumented functions the library
// exports, restricted to the process.
func (p *probe) attachAPI(exe *link.Executable, library string, pid int) []link.Link {
	var links []link.Link
	for cookie, function := range apiFunctions {
		if function.library != library {
			continue
		}
		options := &link.UprobeOptions{PID: pid, Cookie: uint64(cookie)}
		enter, err := exe.Uprobe(function.symbol, p.programs[apiEnterProgram], options)
		if err != nil {
			// Not exported by this library version.
			continue
		}
		exit, err := exe.Uretprobe(function.symbol, p.programs[apiExitProgram], options)
		if err != nil {
			_ = enter.Close()
			continue
		}
		links = append(links, enter, exit)
	}
	return links
}
