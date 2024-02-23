#pragma once
#include <string>
#include <chrono>
#include <iostream>

class MsClock {
public:
	MsClock() {
		reset();
	}

	void reset() {
		startTime = std::chrono::system_clock::now();
	}

	void printAndReset(std::string msg) {
		std::cout << msg << " " << 
			std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime).count() << std::endl;
		reset();
	}

	std::chrono::system_clock::time_point startTime;
};

template<typename T> inline T vecZeros() {
	auto g = T::zeros();
	return T(g.val);
}

template<typename T> inline T vecOnes() {
	auto g = T::ones();
	return T(g.val);
}
