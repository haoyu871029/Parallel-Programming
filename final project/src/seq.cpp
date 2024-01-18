#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <stack>
#include <algorithm>
#include <list>
#include <sstream> 
#include <cmath>
#include <climits>
#include <queue>
#include <algorithm>
#include <time.h>

#include <boost/unordered_map.hpp>
#include <boost/fusion/container/map.hpp>
#include <boost/fusion/include/map.hpp>
using namespace boost;

// #include <unordered_map>
// #include <map>
using namespace std;

 
vector<string> tokenize(string const &str) 
{ 
    vector <string> out; 
    istringstream ss(str);
    string word;
    while (ss >> word) 
    {
        out.push_back(word);
    }
    return out;
}

struct die{
    string techs;
    long double util;
    long double max_area;
    long double area;
};

struct cell{
    string cellName; // for net -> cell info
    string libCellName;
    vector<string> connectNets;
    die* fromBlock, *toBlock;
    long long gain;
    bool locked;
};

long long Fn(cell* c, vector<cell*>& nets){
    long long sum = 0;
    for(auto &it : nets){
        if(it->fromBlock == c->fromBlock){
            sum++;
        } 
    }
    return sum;
}

long long Tn(cell* c, vector<cell*>& nets){
    long long sum = 0;
    for(auto &it : nets){
        if(it->fromBlock == c->toBlock){
            sum++;
        } 
    }
    return sum;
}

long long Gn(cell* c, unordered_map<string, vector<cell*>>& nets){
    // return the gain of this cell
    long long sum = 0;
    for(auto &it : c->connectNets){
        if(Fn(c, nets[it]) == 1){
            sum++;
        }
        if(Tn(c, nets[it]) == 0){
            sum--;
        }
    }
    return sum;
}

bool util_condition(die* td, long double& cellArea){
    // check if the condition is satisfied(add cellArea to die area and compare with max_area)s
    if(((td->area + cellArea) / td->max_area) < td->util){
        return true;
    }
    else{
        return false;
    }
}

void update_gain(cell* c, unordered_map<string, vector<cell*>>& nets, unordered_map<string, unordered_map<string, long double>>& techs, map<long long, list<cell*>>& buckets){

    c->locked = true;

    long long F_n, T_n;

    for(auto it : c->connectNets){
        F_n = Fn(c, nets[it]);
        T_n = Tn(c, nets[it]);

        if(T_n == 0){
            for(auto it2 : nets[it]){
                if(it2->locked == false){
                    buckets[it2->gain].remove(it2);
                    if(buckets[it2->gain].empty()){
                        buckets.erase(it2->gain);
                    }      
                    it2->gain++;
                    buckets[it2->gain].push_back(it2);
                }
            }
        }
        else if(T_n == 1){
            for(auto it2 : nets[it]){
                if(it2->fromBlock == c->toBlock){
                    if(it2->locked == false){
                        buckets[it2->gain].remove(it2);
                        if(buckets[it2->gain].empty()){
                            buckets.erase(it2->gain);
                        }
                        it2->gain--;
                        buckets[it2->gain].push_back(it2);
                    }
                }
            }
        }
    
        F_n = F_n - 1;
    
        if(F_n == 0){
            for(auto it2 : nets[it]){
                if(it2->locked == false){
                    buckets[it2->gain].remove(it2);
                    if(buckets[it2->gain].empty()){
                        buckets.erase(it2->gain);
                    }
                    it2->gain--;
                    buckets[it2->gain].push_back(it2);     
                }
            }
        }
        else if(F_n == 1){
            for(auto it2 : nets[it]){
                if(it2->fromBlock == c->fromBlock){
                    if(it2->locked == false){
                        buckets[it2->gain].remove(it2);
                        if(buckets[it2->gain].empty()){
                            buckets.erase(it2->gain);
                        }
                        it2->gain++;
                        buckets[it2->gain].push_back(it2);
                    }
                }
            }
        }  
    }

    (c->fromBlock)->area -= techs[(c->fromBlock)->techs][c->libCellName];
    (c->toBlock)->area += techs[(c->toBlock)->techs][c->libCellName];
    swap(c->fromBlock, c->toBlock);

}

int cal_cut_size(die &dA, die &dB, unordered_map<string, vector<cell*>>& nets){
    int cut_size = 0;
    for(auto &it : nets){
        bool A = false, B = false;
        for(auto &it2 : it.second){
            if(it2->fromBlock == &dA){
                A = true;
            }
            else if(it2->fromBlock == &dB){
                B = true;
            }
            
        }
        if(A && B){
            cut_size++;
        }
    }
    return cut_size;
}

void Parser(string input_file, string output_file){
    double start_time, end_time;
    start_time = clock();
    // Create a text string, which is used to output the text file
    string myText;

    vector<string> v;
    int NumTechs;

    // Read from the text file
    ifstream MyReadFile(input_file);

    getline(MyReadFile, myText);
    v = tokenize(myText);
    NumTechs = stoi(v[1]);

    unordered_map<string, unordered_map<string, long double>> techs;
    techs.reserve(NumTechs);

    for(int i = 0; i < NumTechs; i++){
        getline (MyReadFile, myText);
        v = tokenize(myText);
        string name = v[1];
        int numCells = stoi(v[2]);

        techs[name].reserve(numCells);

        for(int j = 0; j < numCells; j++){
            getline (MyReadFile, myText);
            v = tokenize(myText);
            string cellName = v[1];
            long double width = stold(v[2]);
            long double height = stold(v[3]);
            techs[name].insert({cellName, width * height});
        }
    }

    
    long double die_width, die_height;

    getline (MyReadFile, myText); // skip the empty line
    getline (MyReadFile, myText);
    v = tokenize(myText);
    die_width = stold(v[1]);
    die_height = stold(v[2]);


    string dA_techs, dB_techs;
    long double dA_max_util, dB_max_util;
    long double die_area = (die_width) * die_height;

    getline (MyReadFile, myText);
    v = tokenize(myText);
    dA_techs = v[1];
    dA_max_util = stold(v[2])/100;

    die dA = {dA_techs, dA_max_util, die_area, 0};
    
    getline (MyReadFile, myText);
    v = tokenize(myText);
    dB_techs = v[1];
    dB_max_util = stold(v[2])/100;

    die dB = {dB_techs, dB_max_util, die_area, 0};

    getline (MyReadFile, myText); // skip the empty line

    int NumCells;

    getline (MyReadFile, myText);
    v = tokenize(myText);
    NumCells = stoi(v[1]);

    unordered_map<string, cell> cells;
    vector<string> cellNames;

    for (int i = 0; i < NumCells; i++)
    {
        getline (MyReadFile, myText);
        v = tokenize(myText);
        string cellName = v[1];
        string libCellName = v[2];
        cells[cellName].cellName = cellName;
        cells[cellName].libCellName = libCellName;
        cells[cellName].locked = false;
        cellNames.push_back(cellName);

    }
    
    getline (MyReadFile, myText); // skip the empty line

    int NumNets;

    getline (MyReadFile, myText);
    v = tokenize(myText);
    NumNets = stoi(v[1]);

    unordered_map<string, vector<cell*>> nets;

    for (int i = 0; i < NumNets; i++)
    {
        getline (MyReadFile, myText);
        v = tokenize(myText);
        string netName = v[1];
        int numCells = stoi(v[2]);
        vector<cell*> netCells;
        for (int j = 0; j < numCells; j++)
        {
            getline (MyReadFile, myText);
            v = tokenize(myText);
            string cellName = v[1];
            netCells.push_back(&cells[cellName]);
            cells[cellName].connectNets.push_back(netName);
        }
        nets.insert({netName, netCells});
    }
    
    // Close the file
    MyReadFile.close();

    // InitialPartition

    for(auto &it : cellNames){
        if(((techs[dA.techs][cells[it].libCellName])  / dA.util ) <= ((techs[dB.techs][cells[it].libCellName]) / dB.util)){
            if(util_condition(&dA, techs[dA.techs][cells[it].libCellName])){
                cells[it].fromBlock = &dA;
                cells[it].toBlock = &dB;
                dA.area += techs[dA.techs][cells[it].libCellName];
            }
            else{
                cells[it].fromBlock = &dB;
                cells[it].toBlock = &dA;
                dB.area += techs[dB.techs][cells[it].libCellName];
            }
        }
        else{
            if(util_condition(&dB, techs[dB.techs][cells[it].libCellName])){
                cells[it].fromBlock = &dB;
                cells[it].toBlock = &dA;
                dB.area += techs[dB.techs][cells[it].libCellName];
            }
            else{
                cells[it].fromBlock = &dA;
                cells[it].toBlock = &dB;
                dA.area += techs[dA.techs][cells[it].libCellName];
            }
        }
    }

    priority_queue<pair<long long, cell*>> pq;
    
    for(auto &it : cells){
       pq.push({Gn(&it.second, nets), &it.second});
    }

    while(!pq.empty()){
        auto it = pq.top();
        pq.pop();
        if(it.first > 0){
            if(util_condition(it.second->toBlock , techs[it.second->toBlock->techs][it.second->libCellName])){
                it.second->fromBlock->area -= techs[it.second->fromBlock->techs][it.second->libCellName];
                it.second->toBlock->area += techs[it.second->toBlock->techs][it.second->libCellName];
                swap(it.second->fromBlock, it.second->toBlock);
            }
        }
    }
    
    // InitialGain
    map<long long, list<cell*>> buckets;

    for(auto it = cells.begin(); it != cells.end(); it++){
        long long gain = Gn(&(it->second), nets);
        buckets[gain].push_back(&it->second);
        it->second.gain = gain; 
    }

    // main loop
    
    long long gain_sum, max_gain_sum;
    long long step, max_step;
    stack<cell*> cellStk; // cell stack for restoring
    while(true){
        gain_sum = 0;
        max_gain_sum = 0;
        step = 0;
        max_step = 0;
        do{

            for (auto it = buckets.rbegin(); it != buckets.rend(); it++) {
                for(auto it2: it->second){
                    if(util_condition(it2->toBlock, techs[it2->toBlock->techs][it2->libCellName])){
                        
                        update_gain(it2, nets, techs, buckets);
                    
                        buckets[it2->gain].remove(it2);
                        if(buckets[it2->gain].empty()){
                            buckets.erase(it2->gain);
                        }

                        cellStk.push(it2);
                        gain_sum += it2->gain;
                        step++;
                        if(gain_sum >= max_gain_sum){
                            max_gain_sum = gain_sum;
                            max_step = step;
                        }

                        goto next;
                    }
                }
            }
            // means no cell can be moved
            break;
            next:
            if(buckets.empty()){
                break;
            }
            if((clock() - start_time) / CLOCKS_PER_SEC >= 280){
                break;
            }

        }while(gain_sum > 0);

        // recover to max gain sum state using stack
        max_step = step - max_step;
        for(long long i = 0; i < max_step; i++){
            cell* c = cellStk.top();
            cellStk.pop();
            c->fromBlock->area -= techs[c->fromBlock->techs][c->libCellName];
            c->toBlock->area += techs[c->toBlock->techs][c->libCellName];
            swap(c->fromBlock, c->toBlock);
        }

        if(step <= 1){
            break;
        }

        if((clock() - start_time) / CLOCKS_PER_SEC >= 280){
            break;
        }

        buckets.clear();
        for(auto it = cells.begin(); it != cells.end(); it++){
            long long gain = Gn(&(it->second), nets);
            buckets[gain].push_back(&it->second);
            it->second.gain = gain; 
            it->second.locked = false;
        }
    }
    
    // output
    // caculate cut size
    long long cut_size = cal_cut_size(dA, dB, nets);

    vector<string> DieA, DieB;
    for(auto &it : cells){
        if(it.second.fromBlock == &dA){
            DieA.push_back(it.second.cellName);
        }
        else{
            DieB.push_back(it.second.cellName);
        }
    }

    // output to file
    ofstream MyWriteFile(output_file);
    MyWriteFile << "CutSize " << cut_size << endl;
    MyWriteFile << "DieA " << DieA.size() << endl;
    for(auto &it : DieA){
        MyWriteFile << it << endl;
    }

    MyWriteFile << "DieB " << DieB.size() << endl;
    for(auto &it : DieB){
        MyWriteFile << it << endl;
    }
    MyWriteFile.close();
    

}

int main(int argc, char** argv) {
    string input_file = argv[1];
    string output_file = argv[2];
    Parser(input_file, output_file);
    return 0;
}

/* compile & execute */

//(copy this code to main.cpp)
//compile in apollo: 
// (using boost) g++ -std=c++11 -O3 -lm -I /usr/local/include/boost/ -o ../bin/main main.cpp
// (not using boost) g++ -std=c++11 -O3 -lm -o ../bin/main main.cpp
//execute in apollo: ../bin/main ../testcase/public5.txt ../output/public5.out

/* verify */

//(cd to verifier)
//chmod 500 verify
//./verify ../testcase/public5.txt ../output/public5.out