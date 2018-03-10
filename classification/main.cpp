#include"NaiveBayes.h"

#include<iostream>

int main()
{
    NaiveBayes nb;
    nb.set_training_data_file(string("train.data"));
    nb.train();

    string class_=nb.classify(string("chinese chinese chinese tokyo japan"));
    cout<<class_<<endl;
    class_=nb.classify(string("tokyo japan"));
    cout<<class_<<endl;
    return 0;
}
