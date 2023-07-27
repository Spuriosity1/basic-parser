#include <basic_parser.hh>
#include <algorithm>
#include <cmath>

using namespace basic_parser;

int main (int argc, const char* argv[]) 
{
    int nreps;
    double g, gp, rho_g, phi_g;
    bool save_extra_data;
    std::string extra_files;

    Parser p(1,0);
    p.declare("nreps", &nreps);

    p.declare("g", &g);
    p.declare("gp", &gp);
    p.declare("phi_g", &phi_g);
    p.declare("rho_g", &rho_g);
    p.declare_optional("save_extra_data", &save_extra_data, false);
    p.declare("extra_files", &extra_files);

    if (p.all_initialised({"g","gp"})){
        if (p.any_initialised({"phig","rhog"})){
            throw std::runtime_error("If (g,g') are defined, neither of (phig, rhog) may be used.\n");
        }
        p.set_value("phi_g", atan2(gp, g));
        p.set_value("rho_g", sqrt(gp*gp + g*g));
    } else if (p.all_initialised({"phig","rhog"})) {
        if (p.any_initialised({"g","g'"})){
            throw std::runtime_error("If (phig,rhog) are defined, neither of (g, g') may be used.\n");
        }
        p.set_value("g", rho_g * cos(phi_g));
        p.set_value("gp", rho_g * sin(phi_g));
    }
    
    p.from_argv(argc, argv);

    p.assert_all_initialised();
}