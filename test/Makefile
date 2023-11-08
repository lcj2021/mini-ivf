CXX = g++
CXXFLAGS = -std=c++11 -I../include

SRC = test_distance.cpp ../src/distance.cpp
OUT = test_distance.out

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(OUT) -march=native

.PHONY: clean
clean:
	rm -f $(OUT)