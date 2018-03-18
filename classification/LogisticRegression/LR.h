#include<iostream>
#include<fstream>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <map>

using namespace std;

class LR
{
public:
    LR();
    ~LR() {};
    void set_training_data_path(const string filepath);
    void set_testing_data_path(const string filepath);
    void get_train_data();
    void get_test_data();
    void train();
    void test();
private:
    vector<string> split(const string &s, char delim);
    double vec_norm(map<int,double>& w1, map<int,double>& w2);
    double sigmoid(double x);
    double classify(map<int,double>& features, map<int,double>& weights);
    
    
    string train_filepath;
    string test_filepath;   

    vector<map<int,double> > data;
    vector<map<int,double> > data_test;
    map<int, double> weights;
    

    //Learning rate
    double alpha;
    //Max iterations
    unsigned int maxit;
    //Shuffle data set
    bool shuf;
    //Convergence threshold
    double eps;
};
