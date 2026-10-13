// SPDX-License-Identifier: Apache-2.0

package cupti // import "go.opentelemetry.io/ebpf-profiler/gpu/cupti"

// apiSymbols is the curated set of cuDNN/cuBLAS entry points instrumented
// with enter/exit span probes. The slice index is the eBPF attach cookie
// (GPUEvent.V1 for EvAPISpan); symbols missing from a library version are
// skipped at attach.
var apiSymbols = []struct {
	lib     string // key into scanLibs output
	symbol  string
	display string
}{
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

// apiSymbolDisplay returns the display name for an EvAPISpan cookie.
func apiSymbolDisplay(idx uint64) string {
	if idx < uint64(len(apiSymbols)) {
		return apiSymbols[idx].display
	}
	return "api:unknown"
}
