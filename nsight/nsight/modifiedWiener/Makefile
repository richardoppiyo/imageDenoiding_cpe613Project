.PHONY: clean run

NVCC=nvcc -lineinfo

CBLAS_DIR=`spack location -i cblas`
BLAS_DIR=`spack location -i blas`

main: main.cu
	${NVCC} -o main \
		-I. -I${CBLAS_DIR}/include \
		main.cu  \
		-L${CBLAS_DIR}/lib \
		-L${BLAS_DIR}/lib \
		-Xlinker -rpath=${BLAS_DIR}/lib \
		-lcblas -lopenblas

clean:
	rm -f main
