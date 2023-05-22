from setuptools import setup, Extension, find_packages

def main():
    setup(name="cbscs",
          version="1.0.0",
          description="Python C API extension for bscs clusters",
          author="Mateusz Cierniak",
          author_email="mat.cierniak@gmail.com",
          packages=find_packages("pnjl"),
          ext_modules=[Extension(
                        "pnjl.thermo.gcp_cluster.cbscs",
                        sources=[
                            "src/pnjl/thermo/gcp_cluster/cbscs.cpp",
                            "src/pnjl/thermo/gcp_pnjl/c_lattice_cut_sea.cpp",
                            "src/pnjl/thermo/c_distributions.cpp",
                            "src/pnjl/c_globals.cpp",
                        ],
                        libraries = ["gsl", "gslcblas"]
                    )])

if __name__ == "__main__":
    main()