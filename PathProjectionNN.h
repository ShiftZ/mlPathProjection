#pragma once

#include <vector>
#include <array>
#include <mlpack.hpp>

using namespace std;

class PathProjectionNN
{
	static constexpr int input_size = 12;
	static constexpr int output_size = 1;

	int nn_input_size, nn_output_size;

	mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization> nn;

	map<int, array<arma::vec2, output_size>> predictions;
	map<double, array<arma::vec2, input_size + output_size>> dyn_samples;
	vector<arma::vec2> path;
	
public:
	PathProjectionNN();

	double Train(const vector<vector<arma::vec2>>& raw_sequences, 
				 const function<void(int, double)>& epoch_callback,
				 const function<void()>& end_optimization);

	vector<arma::vec2> Predict(int points_n, const function<arma::vec2()>& feeder);

	void Add();
	void DynTrain();

	int GetInputSize();

	void Add(auto&& path_points)
	{
		for (auto& p : path_points)
			path.push_back({p[0], p[1]});
		Add();
	}

	void WriteNN(ostream& stream);
	void ReadNN(istream& stream);
};
