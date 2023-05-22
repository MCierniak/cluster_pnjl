#include "c_globals.h"

gsl_function_wrapper::gsl_function_wrapper(double (*func)(double, void*), size_t i):
    w(gsl_integration_workspace_alloc(i)),
    limit(i)
{
    F.function = func;
    gsl_set_error_handler(&gsl_integ_error);
}

gsl_function_wrapper::gsl_function_wrapper(double (*func)(double, void *)):
    w(gsl_integration_workspace_alloc(100000)),
    limit(100000)
{
    F.function = func;
    gsl_set_error_handler(&gsl_integ_error);
}

gsl_function_wrapper::~gsl_function_wrapper()
{
    gsl_integration_workspace_free(w);
}

double gsl_function_wrapper::eval(double x, void *pars)
{
    F.params = pars;
    return GSL_FN_EVAL(&F, x);
}

double gsl_function_wrapper::qag(void *pars, double a, double b, double epsabs, double epsrel, int key)
{
    double result, error;

    F.params = pars;

    gsl_integration_qag(&F, a, b, epsabs, epsrel, limit, key, w, &result, &error);

    return result;
}

double gsl_function_wrapper::qagiu(void *pars, double a, double epsabs, double epsrel)
{
    double result, error;

    F.params = pars;

    gsl_integration_qagiu(&F, a, epsabs, epsrel, limit, w, &result, &error);

    return result;
}

void pbar(double x, double x_min, double x_max, double x_step)
{
	int prog;
	if (x_min == x_max ) prog = 100;
	else prog = (x - x_min)/(x_max - x_min) * 100;
    std::cerr << "\t\015\x1b[K\033[2A\n\n";
	std::cerr << "Progress: ";
	for(double i = 0; i < 100; i += x_step)
	{
		if (i < prog)  std::cerr << "\x1b[48;5;" << std::to_string(41) << "m \x1b[m";
		else 		   std::cerr << "\x1b[48;5;" << std::to_string(60) << "m \x1b[m";
	}
    std::cerr << " " << prog << "/100";
    if (prog == 100) std::cerr << "\n";
}

void pbar(int i, int i_min, int i_max)
{
    int prog;
	if (i_min == i_max ) prog = 100;
	else prog = (i - i_min)/(i_max - i_min) * 100;
    std::cerr << "\t\015\x1b[K\033[2A\n\n";
	std::cerr << "Progress: ";
	for(int j = 0; j < 100; j++)
	{
		if (j < prog)  std::cerr << "\x1b[48;5;" << std::to_string(41) << "m \x1b[m";
		else 		   std::cerr << "\x1b[48;5;" << std::to_string(60) << "m \x1b[m";
	}
    std::cerr << " " << prog << "/100";
    if (prog == 100) std::cerr << "\n";
}

void gsl_integ_error(const char *reason, const char *file, int line, int gsl_errno)
{
	if (gsl_errno == GSL_EROUND) {}
	else if (gsl_errno == GSL_EDIVERGE) {}
	else if (gsl_errno == GSL_EOVRFLW) {}
    else if (gsl_errno == GSL_ESING) {
        // std::cout << "Found singularity!" << std::endl;
    }
	else
	{
		std::cout << "Errno: " << gsl_errno << std::endl;
		throw std::runtime_error("Błąd GSL'a!");
	}
}