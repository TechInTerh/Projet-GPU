all:
	mkdir -p build
	cd build; cmake -DCMAKE_BUILD_TYPE=Release ..
	cd build; make -j4

debug:
	mkdir -p debug
	cd debug; cmake -DCMAKE_BUILD_TYPE=Debug ..
	cd debug; make

clean:
	rm -rf build debug

.PHONY: all clean debug
