#include"LR.h"

#include<iostream>

int main()
{
    LR lr;
    lr.set_training_data_path(string("train.dat"));
    lr.set_testing_data_path(string("test.dat"));

    lr.get_train_data();
    lr.get_test_data();

    lr.train();
    lr.test();

    return 0;
}
