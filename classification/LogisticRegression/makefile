main : main.o LR.o
	g++ -std=c++11 -o main main.o LR.o

main.o : main.cpp LR.h
	g++ -std=c++11 -c main.cpp -lm
LR.o : LR.cpp LR.h
	g++ -std=c++11 -c LR.cpp -lm

clean:
	rm *.o
