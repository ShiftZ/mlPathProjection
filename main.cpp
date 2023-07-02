#include "pch.h"

#include "main.h"

#define AGG_BGR24
#include "PathProjectionNN.h"
#include "agg/examples/pixel_formats.h"
#include "utils.h"

using namespace std;
using namespace std::chrono;
using namespace agg;
using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;

constexpr size_t operator ""_sz ( unsigned long long n ){ return n; }

constexpr int prediction_size = 10;
constexpr double out_coords_scale = 1;
constexpr int training_data_size = 5000;
constexpr bool load_training_data = false;
constexpr bool save_training_data = false;

constexpr int window_width = 1024;
constexpr int window_height = 768;

filesystem::path nn_params_filename = "nn_params";
filesystem::path training_data_filename = "training_data.txt";

the_application::the_application( pix_format_e format )
	: platform_support(format, false), nn(new PathProjectionNN), pf(rbuf_window()), font_cache(font_engine)
{
	ifstream nn_params_file(nn_params_filename, ios::binary);
	if (nn_params_file.is_open())
	{
		nn->ReadNN(nn_params_file);
		trained = true;
	}

	if (load_training_data)
	{
		load_data();
		train();
	}
}

void the_application::on_init()
{
	font_engine.load_font("roboto_medium.ttf", 0, glyph_ren_agg_gray8);
	font_engine.hinting(true);
	font_engine.flip_y(true);
	font_engine.height(24);

	render_base = make_unique<renderer_base<pixfmt_bgr24>>(pf);
}

void the_application::on_draw()
{
	rasterizer_scanline_aa<> ras;
	
	render_base->clear(rgba(1, 1, 1));

	simple_path line(mouse);
	conv_stroke stroke_path(line);

	double stroke_width = 2.0;
	stroke_path.width(stroke_width);
	stroke_path.line_cap(square_cap);
	stroke_path.line_join(miter_join);
	stroke_path.miter_limit(stroke_width);

	ras.reset();
	ras.add_path(stroke_path);

	scanline_p8 sl;
	render_scanlines_aa_solid(ras, sl, *render_base, rgba8(0x22, 0x22, 0x22, 0xff));

	if (!prediction.empty())
	{
		simple_path prediction_line({mouse.back()});
		prediction_line.points.insert(prediction_line.points.end(), prediction.begin(), prediction.end());
		stroke_path.attach(prediction_line);
		ras.add_path(stroke_path);
		render_scanlines_aa_solid(ras, sl, *render_base, rgba8(0, 0xff, 0, 0xff));
	}

	/*if (!prediction_errors.empty())
	{
		dotted_line err_lines(prediction_errors);
		conv_stroke<dotted_line> dotted_stroke(err_lines);
		ras.add_path(dotted_stroke);
		render_scanlines_aa_solid(ras, sl, *render_base, rgba8(0xff, 0, 0, 0xff));
	}*/

	if (!losses.empty())
		draw_text(to_string(losses.back()), 10, 50);

	if (!trained)
		draw_text(to_string(collected_data_size), 10, 50);

	if (prediction_error != 0)
		draw_text("NN projection error: " + to_string(prediction_error), 10, 50);

	if (interpolation_error != 0)
		draw_text("plain projection error: " + to_string(interpolation_error), 10, 80);

	if (!trained && training.valid())
	{
		string training_text = "epoch " + to_string(epoch_n);
		for (double loss : epoch_losses)
			training_text += '\n' + to_string(loss);

		draw_text(training_text, window_width - 120, 20);
	}
	else if (training_set_error != 0)
	{
		draw_text("Training set error: " + to_string(training_set_error), window_width - 300, 20);
	}

	next_update = system_clock::now() + 1666us;
}

void the_application::draw_text( string_view str, double x, double y, rgba8 color )
{
	double initial_x = x;

	for (char ch : str)
	{
		const glyph_cache* glyph = font_cache.glyph(ch);
		if (ch == '\n')
		{
			x = initial_x;
			y += (glyph->bounds.y2 - glyph->bounds.y1) + 4;
		}
		else
		{
			font_cache.init_embedded_adaptors(glyph, x, y);
			render_scanlines_aa_solid(font_cache.gray8_adaptor(), font_cache.gray8_scanline(), *render_base, color);

			x += glyph->advance_x;
			y += glyph->advance_y;

			font_cache.add_kerning(&x, &y);
		}
	}
}

void the_application::on_mouse_move( int x, int y, unsigned flags )
{
	auto now = system_clock::now();

	if (!mouse_times.empty() && now - mouse_times.back() < 6ms) return;
	if (!mouse.empty() && mouse.back()[0] == x && mouse.back()[1] == y) return;

	auto flush_sequence = [&]
	{
		if (mouse.size() >= nn->GetInputSize() && !trained)
			training_data.emplace_back(mouse.begin(), mouse.end());

		mouse.clear(), mouse_times.clear();
		prediction.clear();
		prediction_errors.clear();
	};

	if (mouse.size() >= 3)
	{
		auto it = mouse_times.rbegin();
		duration common_rate1 = *(it) - *(it + 1);
		duration common_rate2 = *(it + 1) - *(it + 2);
		duration common_rate = min(common_rate1, common_rate2);
		if (now - mouse_times.back() > common_rate * 1.8) 
			flush_sequence();
	}

	if (mouse.size() >= 4)
	{
		vec2 interpl_v = mouse.back() + interpl_predict();
		vec2 err_v = interpl_v - vec2{double(x), double(y)};

		interpolation_errors.push_back(length(err_v));
		interpolation_error = ranges::fold_right(interpolation_errors, 0, plus()) / interpolation_errors.size();
		if (prediction_errors.size() > 100) prediction_errors.erase(prediction_errors.begin());
	}

	mouse.push_back({double(x), double(y)});
	mouse_times.push_back(now);

	if (!trained)
	{
		if (mouse.size() > nn->GetInputSize())
			++collected_data_size;

		if (collected_data_size == training_data_size)
		{
			flush_sequence();
			if (save_training_data) save_data();
			train();
		}
	}
	else
	{
		if (!prediction.empty())
		{
			double err = length(mouse.back() - prediction[0]);
			prediction_errors.push_back(err);
			prediction_error = ranges::fold_right(prediction_errors, 0, plus()) / prediction_errors.size();
			if (prediction_errors.size() > 100) prediction_errors.erase(prediction_errors.begin());
		}

		prediction = predict();
	}

	force_redraw();
}

void the_application::on_idle()
{
	if (system_clock::now() > next_update) force_redraw();
}

void the_application::train()
{
	training = async([&]
	{
		auto epoch_callback = [&](int epoch, double loss)
		{
			ranges::move_backward(epoch_losses, epoch_losses + size(epoch_losses) - 1,  epoch_losses + size(epoch_losses));
			epoch_losses[0] = loss;
			epoch_n = epoch;
			force_redraw(); // thread-safe on windows
		};

		training_set_error = nn->Train(training_data, epoch_callback, {});
		trained = true;

		ofstream file(nn_params_filename, ios::binary);
		nn->WriteNN(file);

		force_redraw();
	});
}

vector<vec2> the_application::predict()
{
	if (mouse.size() <= nn->GetInputSize()) return {};
	auto feeder = [it = (mouse.rbegin() + nn->GetInputSize()).base()]() mutable { return *(it++); };
	vector<vec2> predictions = nn->Predict(prediction_size, feeder);
	return predictions;
}

vec2 the_application::interpl_predict()
{
	int end = mouse.size() - 1;
	vec2 v1 = mouse[end] - mouse[end - 1];
	vec2 v2 = mouse[end - 1] - mouse[end - 2];
	vec2 v3 = mouse[end - 2] - mouse[end - 3];

	if (equal(v1, v2) || equal(v2, v3)) return {0, 0};

	double l1, l2;

	v1 = normalized(v1, l1);
	v2 = normalized(v2, l2);

	double a1 = angle(v1, v2);
	double a2 = angle(v2, v3);

	double a0 = a1;// + (a1 - a2);
	double l0 = l1;// + (l1 - l2);

	vec2 v0 = rotate(v1, {cos(a0), sin(a0)});
	v0 = v0 * l0;

	return v0;
}

void the_application::save_data()
{
	ofstream file(training_data_filename, ios::out | ios::trunc);
	for (vector<vec2>& seq : training_data)
	{
		string seq_txt;
		for (vec2& pt : seq)
			seq_txt += std::format("({},{}) ", (int)pt[0], (int)pt[1]);
		file << seq_txt << '\n';
	}
}

void the_application::load_data()
{
	ifstream file(training_data_filename);
	string line;
	regex vec_coords("\\(([0-9]+),([0-9]+)\\)");

	while (getline(file, line))
	{
		vector<vec2> seq;

		smatch matches;
		for (auto it = line.cbegin(); regex_search(it, line.cend(), matches, vec_coords); it = matches.suffix().first)
		{
			vec2& pt = seq.emplace_back();
			from_chars(&*matches[1].first, &*matches[1].second, pt[0]);
			from_chars(&*matches[2].first, &*matches[2].second, pt[1]);
		}

		if (!seq.empty())
			training_data.emplace_back(move(seq));
	}
}

int agg_main( int argc, char* argv[] )
{
	the_application app(pix_format);
	app.caption("ml for fun");
	app.init(window_width, window_height, 0);
	return app.run();
}