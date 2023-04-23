# cff_parse

A **C**rappy **F**ile **F**ormat for scientific input plaintext files.

Many scientists need to manage very small files full of parameters, but often, it's easier to just
throw a million command line arguments into the program and hope for the best. This often does the
job, but creates a nightmare for anyone else reading the code.

This parser has three design principles:
1. Work as a drop-in replacement for command line arguments.
2. Be stateless as much as possible: This class binds 'live' variables from a larger scope to string 'handles', rather than storing the parameters internally.
3. Drag and drop installation. (header only)

This is obviously less capable than a "real" heterogeneous parser, e.g. for TOML, JSON or YAML. In
some sense that's the point: the generality of those formats makes accessing their members in a
strongly typed language syntactically awkward.

Input files look like

```
# this is a comment
system size = 4
T_hot (Kelvin) = 5.11
# the comment can go anywhere
prefix = sys1_
```

The calling program might look like

```c++
#include <basic_parse.hh>
#include <string>

//                  the unsigned format you're using (defaults to uint64_t)
//                  |         the int format you're using (default: int64_t)
//                  |         |    the float format you're using (default: double)
typedef basic_parse<unsigned, int, double> parser_t;

// usage: test infile outfile
int main(int argc, char** argv){

	double T_hot;
	unsigned n;
	std::string pref;
	double T_cold=0;


	parser_t p;

	p.declare("T_hot",&T_hot);
	p.declare_optional("T_cold,&T_cold,0);
	p.declare("system size",&s);
	p.declare("prefix", &pref);
	p.from_file(argv[1]);

	// ... code
	
	// Save elsewhere
	p.into_file(argv[2]);
	return 0;
}
```

# TODO

 - Proper docs
 - Examples
 - Tests
 - python parser
 - FORTRAN parser
 - Julia parser
 - Maybe accept array data. _maybe_.

