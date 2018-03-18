#include"NaiveBayes.h"

#include<float.h>
#include<math.h>

#include<algorithm>
#include<fstream>
#include<iostream>
#include<set>
#include<sstream>

void NaiveBayes::set_training_data_file(const string filepath)
{
    training_data_filepath = filepath;
}

void NaiveBayes::get_training_data()
{
    ifstream file(training_data_filepath.c_str());
    string line;
    while(getline(file,line))
    {
        istringstream iss(line);
        string attr;
        StringVector sv;
        while(getline(iss, attr, DELIM))
        {
            sv.push_back(attr);
        }
        classes.push_back(sv.back()); // The last entry is the class name
        sv.pop_back();
        inputdata.push_back(sv);
    }
}

void NaiveBayes::train()
{
    get_training_data();

    //compute class prob
    for(StringVector::const_iterator it=classes.begin();it!=classes.end();it++)
    {
        if(class_prob.find(*it)==class_prob.end())
        {
            class_prob[*it] = 1.0;
        }
        else
        {
            class_prob[*it] += 1.0;
        }
    }

    for(MapSD::const_iterator it=class_prob.begin();it!=class_prob.end();it++)
    {
        class_prob[it->first] = (it->second) / classes.size();
    }

    /*
     * count is a map of maps.
     * C1: count(x1), ... , count(xn)
     * .
     * CN: count(x1), ... , count(xn)
     */
    MapSD vob;
    vobsize = 0;
    for(int i=0;i<inputdata.size();i++)
    {
        StringVector instance=inputdata[i];

        for(int j=0;j<instance.size();j++)
        {
            if(vob.find(instance[j])==vob.end())
            {
                vob[instance[j]] = 1.0;
                vobsize ++;
            }

            if(count.find(classes[i])==count.end())
            {
                count[classes[i]] = new MapSD;
                (*count[classes[i]])[instance[j]] = 1.0;
            }
            else
            {
               if((*count[classes[i]]).find(instance[j])==(*count[classes[i]]).end()) 
               {
                    (*count[classes[i]])[instance[j]] = 1.0;                    
               }
               else
               {
                    (*count[classes[i]])[instance[j]] += 1.0;                 
               }
            }
        }
    }

    for(MapOfMaps::const_iterator it=count.begin();it!=count.end();it++)
    {
        num_of_words_per_class[it->first] = 1.0;
        for(MapSD::const_iterator iit=it->second->begin();iit!=it->second->end();iit++)
        {
            num_of_words_per_class[it->first] += iit->second;

        }
    }


}


string NaiveBayes::classify(const string& input_attr)
{
    istringstream iss(input_attr);
    string token;
    StringVector attr;
    int i=0;
    while(getline(iss,token,DELIM))
    {
        attr.push_back(token);
    }

    double max_prob = -DBL_MAX;
    string max_class;

    for(MapOfMaps::const_iterator it=count.begin();it!=count.end();it++)
    {
        double prob=0.0;
        for(int i=0;i<attr.size();i++)
        {
            double p;
            if(it->second->find(attr[i])==it->second->end())
            {// add 1 smoothing
                p= (0+1.0)/(num_of_words_per_class[it->first]+vobsize);
            }
            else
            {
                p= ((*it->second)[attr[i]]+1.0)/(num_of_words_per_class[it->first]+vobsize);
                
            }
            prob += log(p);
        }
        prob += log(class_prob[it->first]);
        if(prob>max_prob)
        {
            max_prob=prob;
            max_class = it->first;
        }
    }
    
    return max_class;
}

