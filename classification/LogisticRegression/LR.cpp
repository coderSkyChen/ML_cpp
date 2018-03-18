#include"LR.h"
#include<stdio.h>

LR::LR()
{
    //Learning rate
    alpha=0.001;
    //Max iterations
    maxit=200;
    //Shuffle data set
    shuf=true;
    //Convergence threshold
    eps=0.005;
}

void LR::set_training_data_path(const string filepath)
{
    train_filepath=filepath;
}

void LR::set_testing_data_path(const string filepath)
{
    test_filepath=filepath;
}

void LR::train()
{
    random_device rd;
    mt19937 g(rd());

    vector<int> index(data.size());
    iota(index.begin(),index.end(),0); //0,1,2,...

    double norm=1.0;
    int n=0;

    cout<<"# sgd begining~"<<endl;
    while(norm>eps)
    {
        map<int,double> old_weights(weights);
        if(shuf)    shuffle(index.begin(),index.end(),g);

        for(int i=0;i<data.size();i++)
        {
            int label=data[index[i]][0];
            double predicted = classify(data[index[i]],weights);
            for(auto it=data[index[i]].begin();it!=data[index[i]].end();it++)
            {
                if(it->first!=0)
                {
                    weights[it->first] += alpha*(label-predicted)*it->second;
                }
            }
        }
        norm=vec_norm(weights,old_weights);
        if(n&&n%20==0)
        {
            printf("# convergence: %1.4f iterations: %i\n",norm,n);
        }

        if(++n>maxit)
            break;

    }
}

void LR::test()
{
    double tp=0,tn=0,fp=0,fn=0;
    
    for(int i=0;i<data_test.size();i++)
    {
        int label=data_test[i][0];
        double predicted = classify(data_test[i],weights);

        if(((label==-1||label==0)&&predicted<0.5)||(label==1&&predicted>=0.5))
        {
            if(label==1) tp++;  else tn++;            
        }
        else
        {
            if(label==1) fn++;  else fp++;
        }

  }
   printf ("# accuracy:    %1.4f (%i/%i)\n",((tp+tn)/(tp+tn+fp+fn)),(int)(tp+tn),(int)(tp+tn+fp+fn));
   printf ("# precision:   %1.4f\n",tp/(tp+fp));
   printf ("# recall:      %1.4f\n",tp/(tp+fn));
   printf ("# tp:          %i\n",(int)tp);
   printf ("# tn:          %i\n",(int)tn);
   printf ("# fp:          %i\n",(int)fp);    
   printf ("# fn:          %i\n",(int)fn);
}

void LR::get_train_data()
{
    cout<<"# loading traindata..."<<endl;
    random_device rd;
    mt19937 g(rd());

    ifstream fin;
    string line;


    fin.open(train_filepath);
    while(getline(fin,line))
    {
        if(line.length())        
        {
            if(line[0]!='#'&&line[0]!=' ')
            {
                vector<string> tokens=split(line,' ');
                map<int,double> example;
                if(atoi(tokens[0].c_str())==1)
                    example[0]=1;
                else
                    example[0]=0;

                for(int i=0;i<tokens.size();i++)
                {
                    vector<string> feat_val=split(tokens[i],':');
                    if(feat_val.size()==2)
                    {
                        example[atoi(feat_val[0].c_str())]=atof(feat_val[1].c_str());
                        weights[atoi(feat_val[0].c_str())] = -1.0+2.0*(double)rd()/rd.max();  //-1 ~1
                    }
                }
                data.push_back(example);
            }
        }
    }
    fin.close();
    cout << "# training examples: " << data.size() << endl;
    cout << "# features:          " << weights.size() << endl;
}
void LR::get_test_data()
{
    cout<<"# loading testdata..."<<endl;

    ifstream fin;
    string line;


    fin.open(test_filepath);
    while(getline(fin,line))
    {
        if(line.length())        
        {
            if(line[0]!='#'&&line[0]!=' ')
            {
                vector<string> tokens=split(line,' ');
                map<int,double> example;
                if(atoi(tokens[0].c_str())==1)
                    example[0]=1;
                else
                    example[0]=0;

                for(int i=0;i<tokens.size();i++)
                {
                    vector<string> feat_val=split(tokens[i],':');
                    if(feat_val.size()==2)
                    {
                        example[atoi(feat_val[0].c_str())]=atof(feat_val[1].c_str());
                    }
                }
                data_test.push_back(example);
            }
        }
    }
    fin.close();
    cout << "# testing examples: " << data_test.size() << endl;
}

vector<string> LR::split(const string &s, char delim)
{
    vector<string> elems;
    stringstream ss(s);
    string item;
    while(getline(ss,item,delim))
    {
        elems.push_back(item);
    }
    return elems;
}

double LR::vec_norm(map<int,double>& w1, map<int,double>& w2)
{
     double sum = 0.0;
     for(auto it = w1.begin(); it != w1.end(); it++){
         double minus = w1[it->first] - w2[it->first];
         double r = minus * minus;
         sum += r;
    }
    return sqrt(sum);
}

double LR::sigmoid(double x){
    static double overflow = 20.0;
    if (x > overflow) x = overflow;
    if (x < -overflow) x = -overflow;

    return 1.0/(1.0 + exp(-x));
}

double LR::classify(map<int,double>& features, map<int,double>& weights){

    double logit = 0.0;
    for(auto it = features.begin(); it != features.end(); it++){
        if(it->first != 0){
            logit += it->second * weights[it->first];
        }
    }
    return sigmoid(logit);
}
