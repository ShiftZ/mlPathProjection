#pragma once

using namespace agg;
using namespace std;
using namespace chrono;
using namespace arma;
using namespace mlpack;

class the_application : public platform_support
{
	unique_ptr<class PathProjectionNN> nn;

	vector<double> losses;

	bool trained = false;
	int collected_data_size = 0;
	double training_set_error = 0;

	vector<vec2> mouse, prediction;

	vector<double> prediction_errors;
	double prediction_error = 0;

	vector<double> interpolation_errors;
	double interpolation_error = 0;

	vector<system_clock::time_point> mouse_times;
	vector<vector<vec2>> training_data;

	pixfmt_bgr24 pf;
	unique_ptr<renderer_base<pixfmt_bgr24>> render_base;

	font_engine_freetype_int32 font_engine;
	font_cache_manager<font_engine_freetype_int32> font_cache;

	system_clock::time_point next_update;

	future<void> training;
	int epoch_n = 0;
	double epoch_losses[15] = {};

public:
	the_application( pix_format_e format );
	void on_init() override;
	void on_draw() override;
	void on_mouse_move( int x, int y, unsigned flags ) override;
	void on_idle() override;

	void draw_text( string_view str, double x, double y, rgba8 color = {0, 0, 0, 0xff} );
	
	void train();
	vector<vec2> predict();
	vec2 interpl_predict();
	void save_data();
	void load_data();
};