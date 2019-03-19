#include <fstream>
#include <sstream>
#include <cctype>
#include <algorithm>
#include <unordered_map>

#include "util.h"

// ------------------------ Read CSV file -------------------------------------------------

std::string trim(const std::string &s) {
  auto wsfront=std::find_if_not(s.begin(),s.end(),[](int c){return std::isspace(c);});
  auto wsback=std::find_if_not(s.rbegin(),s.rend(),[](int c){return std::isspace(c);}).base();
  return (wsback<=wsfront ? std::string() : std::string(wsfront,wsback));
}

void read_csv(string filename, vector<Example>& result) {
  ifstream file(filename);
  string line;

  while (getline(file, line)) {
    Example e;
    istringstream s(line);
    string field;

    while (getline(s, field, ';')) {
      if (s.peek() != '\n' && s.peek() != EOF)
        e.traits.push_back(stod(field));
    }

    e.category = trim(field);
    e.n = e.traits.size();
    result.push_back(e);
  }
}

// -------------------- Divide data into partitions ---------------------------------------------

vector<vector<Example>> make_partitions(const vector<Example>& data) {
  unordered_map<string, int> category_count;
  vector<vector<Example>> partitions;

  for (int i = 0; i < K; i++)
    partitions.push_back(vector<Example>());

  for (auto& ex : data) {
    int count = category_count[ex.category];
    partitions[count].push_back(ex);

    category_count[ex.category] = (category_count[ex.category] + 1) % K;
  }

  return partitions;
}

// ------------------------------ Functions on data points ---------------------------------------

double distance_sq(const Example& e1, const Example& e2) {
  double distance = 0.0;
  for (int i = 0; i < e1.n; i++)
    distance += (e2.traits[i] - e1.traits[i]) * (e2.traits[i] - e1.traits[i]);
  return distance;
}

double distance_sq_weights(const Example& e1, const Example& e2, const vector<double>& w) {
  double distance = 0.0;
  for (int i = 0; i < e1.n; i++)
    distance += w[i] * (e2.traits[i] - e1.traits[i]) * (e2.traits[i] - e1.traits[i]);
  return distance;
}
