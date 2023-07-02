#pragma once

#include <agg_basics.h>
#include <armadillo>

using namespace std;
using namespace agg;
using namespace arma;

struct simple_path
{
	vector<vec2> points;
	int pos = 0;

	simple_path( vector<vec2> points = {} ) : points(move(points)) {}

	void rewind( int = 0 ) { pos = 0; }

	unsigned vertex( double* x, double* y )
	{
		if (pos >= points.size()) return path_cmd_stop;
		*x = points[pos][0], *y = points[pos][1];
		return (pos++ == 0) ? path_cmd_move_to : path_cmd_line_to;
	}
};

struct dotted_line
{
	vector<pair<vec2, vec2>> lines;
	int pos = 0;

	dotted_line( vector<pair<vec2, vec2>> lines ) : lines(move(lines)) {}

	void rewind( int = 0 ) { pos = 0; }

	unsigned vertex( double* x, double* y )
	{
		int line_n = pos / 2;
		if (line_n >= lines.size()) return path_cmd_stop;

		if (pos++ % 2 == 0)
		{
			*x = lines[line_n].first[0];
			*y = lines[line_n].first[1];
			return path_cmd_move_to;
		}
		else
		{
			*x = lines[line_n].second[0];
			*y = lines[line_n].second[1];
			return path_cmd_line_to;
		}
	}
};

inline double length( const vec2& v ) { return sqrt(dot(v, v)); }

inline double cross( const vec2& v1, const vec2& v2 )
{
	return v1[1] * v2[0] - v1[0] * v2[1];
}

template< typename type = double >
vec2 normalized( const vec2& v, type&& len = double() )
{
	len = length(v);
	double k = 1. / len;
	return {v[0] * k, v[1] * k};
}

inline double angle( const vec2& v1, const vec2& v2 )
{
	double d = dot(v1, v2);
	double c = cross(v1, v2);
	return atan2(c, d);
}

inline vec2 rotate( const vec2& v, const vec2& rot )
{
	return {v[0] * rot[0] - v[1] * rot[1], v[1] * rot[0] + v[0] * rot[1]};
}

inline bool equal( const vec2& v1, const vec2& v2 )
{
	return v1[0] == v2[0] && v1[1] == v2[1];
}
