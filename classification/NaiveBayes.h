#include<map>
#include<string>
#include<vector>
using namespace std;

typedef vector<string> StringVector;
typedef vector<StringVector> TwoDVector;
typedef map<string, double> MapSD;
typedef map<string, MapSD*> MapOfMaps;

const char DELIM = ' ';

class NaiveBayes
{
public:
    NaiveBayes() {};
    ~NaiveBayes() {};
    void set_training_data_file(const string filepath);
    void train();
    string classify(const string& input_attr );

private:
    void get_training_data();
    
    string training_data_filepath;
    StringVector classes; //class label for each input instance
    TwoDVector inputdata;
    MapSD num_of_words_per_class;
    MapSD class_prob;    //p(ci)
    MapOfMaps count;   
    int vobsize; //vob size used in laplace smoothing



};
