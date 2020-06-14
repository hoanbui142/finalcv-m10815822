#include "finalcv-m10815822.hpp"

int main(int argc, char const *argv[])
{
    Reconstruct3D reconstruct3d;
    reconstruct3d.readParams();
    reconstruct3d.printinfo();
    reconstruct3d.checkpoint();

    return 0;
}