#ifdef MACHINELEARNING_DLL_EXPORT
#define MACHINELEARNING_DLL_API __declspec(dllexport) 
#else
#define MACHINELEARNING_DLL_API __declspec(dllimport) 
#endif

extern "C" 
{
	MACHINELEARNING_DLL_API double* LinearCreateModel(int inputDimensions);
	MACHINELEARNING_DLL_API double* LinearRemoveModel(double* model);
	MACHINELEARNING_DLL_API double* LinearFitRegression(double* model, double* inputs, int inputsSize, int inputSize, double* outputs, int outputSize);
	MACHINELEARNING_DLL_API double* LinearFitClassificationClassic(double* model, double* inputs, int inputsSize, int inputSize, double* outputs, int outputSize);
	MACHINELEARNING_DLL_API double* LinearFitClassificationHebb(double* model, double* inputs, int inputsSize, int inputSize, double* outputs, int outputSize);
	MACHINELEARNING_DLL_API double* LinearClassify(double* model, double* inputs, int inputsSize);
	MACHINELEARNING_DLL_API double* LinearPredict(double* model, double* inputs, int inputsSize);
	MACHINELEARNING_DLL_API double LinearRegression(double* model, double* inputs, int inputsSize);

	MACHINELEARNING_DLL_API double*** LinearCreateMLPModel(int nbCouches, int* nbNeurones, int nbInputs);
	MACHINELEARNING_DLL_API double* LinearFitClassificationMulti(double* model, double* inputs, int inputsSize, int inputSize, double* outputs, int outputSize);
}