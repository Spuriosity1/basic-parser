#include <basic_parser.hh>
#include <algorithm>
#include <cmath>

using namespace basic_parser;

int main (char argc, char* argv[]) 
{
    Parser p(1,0);
    p.declare<int>("nreps");
    p.declare<std::string>("label");
    p.declare<std::filesystem::path>("infile1");
    p.declare<std::filesystem::path>("output/outdata");

    p.declare<double>("g");
    p.declare<double>("gp");
    p.declare<double>("phig");
    p.declare<double>("rhog");

    if (p.all_initialised({"g","gp"})){
        if (p.any_initialised({"phig","rhog"})){
            throw std::runtime_error("If (g,g') are defined, neither of (phig, rhog) may be used.\n");
        }
        p.get<double>("phig") = atan2(p.get<double>("gp"), p.get<double>("g"));
        p.get<double>("rhog") = sqrt(p.get<double>("gp")*p.get<double>("gp") + p.get<double>("g")*p.get<double>("g"));
    } else if (p.all_initialised({"phig","rhog"})) {
        if (p.any_initialised({"g","g'"})){
            throw std::runtime_error("If (phig,rhog) are defined, neither of (g, g') may be used.\n");
        }
        p.get<double>("g") = p.get<double>("rhog") * cos(p.get<double>("phig"));
        p.get<double>("gp") = p.get<double>("rhog") * sin(p.get<double>("phig"));
    }


    p.assert_all_initialised();
}