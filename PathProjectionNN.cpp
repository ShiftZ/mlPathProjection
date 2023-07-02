#include "pch.h"

#include "PathProjectionNN.h"

using namespace arma;
using namespace mlpack;

enum class Mode { Points, Vectors, AnglesLengths };

static constexpr Mode mode = Mode::AnglesLengths;

static constexpr double opt_step = 0.0004;
static constexpr int batch_size = 400;
static constexpr int max_iterations = 5000000;
static constexpr double beta1 = 0.9;
static constexpr double beta2 = 0.999999;

static constexpr double coords_scale = 0.1;
static constexpr double angle_scale = 10;

static constexpr int dynamic_training_samples_n = max(1000, batch_size);

static double length( const vec2& v ) { return sqrt(dot(v, v)); }
static double cross( const vec2& v1, const vec2& v2 ) { return v1[1] * v2[0] - v1[0] * v2[1]; }
static bool equal( const vec2& v1, const vec2& v2 ) { return v1[0] == v2[0] && v1[1] == v2[1]; }

template< typename type = double >
vec2 normalized( const vec2& v, type&& len = double() )
{
	len = length(v);
	double k = 1. / len;
	return {v[0] * k, v[1] * k};
}

static double angle( const vec2& v1, const vec2& v2 )
{
	double d = dot(v1, v2);
	double c = cross(v1, v2);
	return atan2(c, d);
}

static vec2 rotate( const vec2& v, const vec2& rot )
{
	return {v[0] * rot[0] - v[1] * rot[1], v[1] * rot[0] + v[0] * rot[1]};
}

template<Mode> struct TConverter;

template<>
struct TConverter<Mode::Points>
{
	vec2 p0;
	function<vec2()> next;
	TConverter(function<vec2()>&& next) : next(move(next)) { p0 = this->next(); }

	void in(subview_col<double>&& col)
	{
		for (int i = 0; i < col.n_rows;)
		{
			vec2 v = p0 - next();
			col[i++] = v[0], col[i++] = v[1];
		}
	}

	void out(const vec& output, auto&& it)
	{
		for (int i = 0; i < output.n_rows; i += 2)
			*it = p0 + vec2{output[i], output[i + 1]};
	}
};

template<>
struct TConverter<Mode::Vectors>
{
	vec2 p0;
	function<vec2()> next;
	TConverter(function<vec2()>&& next) : next(move(next)) { p0 = this->next(); }

	void in(subview_col<double>&& col)
	{
		for (int i = 0; i < col.n_rows;)
		{
			vec2 p1 = next();
			vec2 v = (p1 - p0) * coords_scale;
			p0 = p1;
			col[i++] = v[0], col[i++] = v[1];
		}
	}

	void out(const vec& output, auto&& it)
	{
		for (int i = 0; i < output.n_rows; i += 2)
			*it = (p0 += vec2{output[i], output[i + 1]} / coords_scale);
	}
};

template<>
struct TConverter<Mode::AnglesLengths>
{
	vec2 p0, p1;
	function<vec2()> next;
	TConverter(function<vec2()>&& next) : next(move(next)) { p0 = this->next(), p1 = this->next(); }
	
	void in(subview_col<double>&& col)
	{
		for (int i = 0; i < col.n_rows;)
		{
			vec2 p2 = next();
			vec2 v1 = p1 - p0;
			vec2 v2 = p2 - p1;
			p0 = p1, p1 = p2;

			double l1, l2;
			v1 = normalized(v1, l1);
			v2 - normalized(v2, l2);

			col[i++] = angle(v2, v1) * angle_scale;
			col[i++] = l2 * coords_scale;
		}
	}

	void out(const vec& output, auto&& it)
	{
		for (int i = 0; i < output.n_rows; i += 2)
		{
			double angle = output[i] / angle_scale;
			vec2 v2 = rotate(normalized(p1 - p0), vec2{cos(angle), sin(angle)}) * (output [i + 1] / coords_scale);
			p0 = p1;
			p1 += v2;
			*it = p1;
		}
	}
};

using Pipe = TConverter<mode>;

struct OptimizationCallbacks
{
	const function<void(int, double)>& end_epoch;
	const function<void()>& end_optimization;

	void EndEpoch(auto& opt, auto& func, const mat& coords, size_t epoch, double loss)
		{ if (end_epoch) end_epoch(epoch, loss); }

	void EndOptimization(auto& opt, auto& func, mat& coords)
		{ if (end_optimization) end_optimization(); }
};

PathProjectionNN::PathProjectionNN()
{
	switch (mode)
	{
	case Mode::Points: nn_input_size = input_size - 1; break;
	case Mode::Vectors: nn_input_size = input_size - 1; break;
	case Mode::AnglesLengths: nn_input_size = input_size - 2; break;
	}

	nn_output_size = output_size;

	nn.Add<Linear>(nn_input_size * 2);
	nn.Add<TanH>();
	nn.Add<Linear>(nn_input_size * 2);
	nn.Add<TanH>();
	nn.Add<Linear>(nn_output_size * 2);
}

double PathProjectionNN::Train(const vector<vector<vec2>>& raw_sequences, 
							   const function<void(int, double)>& epoch_callback,
							   const function<void()>& end_optimization)
{
	int sample_length = input_size + output_size;

	int samples_n = ranges::fold_left(raw_sequences, 0, 
		[&](int n, const vector<vec2>& seq){ return n + max(0ull, seq.size() - sample_length); });

	mat input(nn_input_size * 2, samples_n);
	mat output(nn_output_size * 2, samples_n);

	for (int sample_i = 0; const vector<vec2>& seq : raw_sequences)
	for (int i = 0; i < int(seq.size() - sample_length); ++i, ++sample_i)
	{
		Pipe pipe([it = seq.begin() + i]() mutable { return *(it++); });
		pipe.in(input.col(sample_i));
		pipe.in(output.col(sample_i));
	}

	/*double max_x = 0, max_y = 0;
	for (int i = 0; i < output.n_cols; i++)
	{
		subview_col col = output.col(i);
		max_x += 0.5 * abs(col[0]) + 0.5 * (max_x / (i > 0 ? i : 1));
		max_y += 0.5 * col[1] + 0.5 * (max_y / (i > 0 ? i : 1));
	}

	max_x /= output.n_cols;
	max_y /= output.n_cols;*/

	ens::OptimisticAdam optimizer;
	optimizer.StepSize() = opt_step;
	optimizer.BatchSize() = batch_size;
	optimizer.MaxIterations() = max_iterations;
	optimizer.Beta1() = beta1;
	optimizer.Beta2() = beta2;

	nn.Train(input, output, optimizer, OptimizationCallbacks{epoch_callback, end_optimization});

	double training_samples_error = 0;
	int error_n = 0;

	for (const vector<vec2>& seq : raw_sequences)
	for (auto it = seq.begin(); it < seq.end() - sample_length; ++it)
	{
		vector<vec2> prediction = Predict(output_size, [it = it]() mutable { return *it++; });
		for (int j = 0; j < prediction.size(); j++)
		{
			double error = length(*(it + input_size + j) - prediction[j]);
			training_samples_error += error;
			error_n++;

			copy_n(it, sample_length, dyn_samples[error].begin());
			if (dyn_samples.size() > dynamic_training_samples_n)
				dyn_samples.erase(dyn_samples.begin());
		}
	}

	training_samples_error /= error_n;

	return training_samples_error;
}

vector<vec2> PathProjectionNN::Predict(int points_n, const function<vec2()>& feeder)
{
	vector<vec2> points;
	points.reserve(input_size + (points_n / output_size + 1) * output_size);

	for (int i = 0; i < input_size; i++)
		points.push_back(feeder());

	vec input(nn_input_size * 2);
	vec output(nn_output_size * 2);

	for (int i = 0; i < points_n; i += output_size)
	{
		Pipe pipe([it = points.begin() + i]() mutable { return *(it++); });
		pipe.in(input.col(0));
		nn.Predict(input, output);
		pipe.out(output, back_insert_iterator(points));
	}

	points.erase(points.begin(), points.begin() + input_size);
	return points;
}

void PathProjectionNN::Add()
{
	for (auto it = predictions.begin(); it != predictions.end();)
	{
		auto& [point_id, pred_path] = *it;
		if (point_id > path.size() + output_size)
		{
			double error = 0;

			ranges::subrange real_path(&path[point_id], &path[point_id] + output_size);
			for (auto [pred_pt, real_pt] : views::zip(pred_path, real_path))
				error += length(real_pt - pred_pt) / output_size;

			if (error > dyn_samples.begin()->first)
			{
				copy_n(&path[point_id - input_size], input_size + output_size, dyn_samples[error].data());
				if (dyn_samples.size() > dynamic_training_samples_n)
					dyn_samples.erase(dyn_samples.begin());

				predictions.erase(it++);
			}
			else
				++it;
		}
	}
}

void PathProjectionNN::DynTrain()
{
	int sample_length = input_size + output_size;

	int samples_n = dyn_samples.size();
	auto samples = ranges::subrange(dyn_samples.begin(), dyn_samples.end());

	mat input(nn_input_size * 2, samples_n);
	mat output(nn_output_size * 2, samples_n);

	for (int sample_i = 0; auto& seq : views::values(samples))
	{
		Pipe pipe([it = seq.begin()]() mutable { return *(it++); });
		pipe.in(input.col(sample_i));
		pipe.in(output.col(sample_i));
	}

	//nn.Train(input, output, optimizer);
}

int PathProjectionNN::GetInputSize()
{
	return input_size;
}

void PathProjectionNN::WriteNN(ostream& stream)
{
	nn.Parameters().save(stream);
}

void PathProjectionNN::ReadNN(istream& stream)
{
	nn.Parameters().load(stream);
}