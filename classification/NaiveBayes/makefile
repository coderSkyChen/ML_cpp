main : main.o NaiveBayes.o
	g++ -o main main.o NaiveBayes.o

main.o : main.cpp NaiveBayes.h
	g++ -c main.cpp -lm
NaiveBayes.o : NaiveBayes.cpp NaiveBayes.h
	g++ -c NaiveBayes.cpp -lm

clean:
	rm *.o
