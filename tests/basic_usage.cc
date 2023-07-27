#include <basic_parser.hh>
#include <algorithm>
#include <cmath>

using namespace basic_parser;

int main (char argc, char* argv[]) 
{
    int nreps;
    bool g,gp, rho_g, phi_g;

    Parser p(1,0);
    p.declare("nreps", &nreps);

    p.declare("g", &g);
    p.declare("gp", &gp);
    p.declare("phig", &phi_g);
    p.declare("rhog", &rho_g);

    if (p.all_initialised({"g","gp"})){
        if (p.any_initialised({"phig","rhog"})){
            throw std::runtime_error("If (g,g') are defined, neither of (phig, rhog) may be used.\n");
        }
        phi_g = atan2(gp, g);
        rho_g = sqrt(gp*gp + g*g);
    } else if (p.all_initialised({"phig","rhog"})) {
        if (p.any_initialised({"g","g'"})){
            throw std::runtime_error("If (phig,rhog) are defined, neither of (g, g') may be used.\n");
        }
        g = rho_g * cos(phi_g);
        gp = rho_g * sin(phi_g);
    }


    p.assert_all_initialised();
}