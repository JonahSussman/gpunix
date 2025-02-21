NVCC_FLAGS =  -diag-suppress 186
NVCC_FLAGS += -diag-suppress 68

cuda-rv32ima: src/mini-rv32ima.cu src/mini-rv32ima.h src/default64mbdtc.h assets/DownloadedImage 
	mkdir -p bin
	# nvcc -o bin/$@ $< -g -G -O0 $(NVCC_FLAGS)
	nvcc -o bin/$@ $< -O3 $(NVCC_FLAGS)

assets/DownloadedImage:
	wget https://github.com/cnlohr/mini-rv32ima-images/raw/master/images/linux-6.1.14-rv32nommu-cnl-1.zip -O linux-6.1.14-rv32nommu-cnl-1.zip
	unzip linux-6.1.14-rv32nommu-cnl-1.zip
	mkdir -p assets
	mv Image assets/DownloadedImage
	rm linux-6.1.14-rv32nommu-cnl-1.zip

clean:
	rm -rf bin
	rm -rf assets