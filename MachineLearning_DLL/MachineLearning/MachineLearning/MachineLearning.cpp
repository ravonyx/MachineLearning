#include "MachineLearning.h"
#include <random>
#include <Eigen\Dense>
#include <iostream>
#include <Windows.h>

using namespace Eigen;

extern "C" 
{
	double appro = 0.1;

	double* LinearCreateModel(int inputDimensions)
	{
		double* model = new double[inputDimensions + 1];
		for (int i = 0; i < inputDimensions + 1; i++)
		{
			model[i] = rand() % 10 - 5;
		}

		return model;
	}

	double* LinearRemoveModel(double* model)
	{
		return nullptr;
	}

	double* LinearFitRegression(double* model, double* inputs, int inputsSize, int inputSize, double* outputs, int outputSize)
	{
		MatrixXd X(inputsSize, inputSize + 1);
		MatrixXd Y(inputsSize, outputSize);
		
		for (int i = 0; i < inputsSize; i++)
		{
			X(i, 0) = 1;
			X(i, 1) = inputs[i * inputSize];
			X(i, 2) = inputs[i * inputSize + 1];
		}
		for (int i = 0; i < inputsSize; i++)
		{
			Y(i, 0) = outputs[i];
		}
		
		MatrixXd X1 = (X.transpose() * X);
		MatrixXd W = (X1.inverse() * X.transpose()) * Y;

		for (int i = 0; i <= inputSize; i++)
			model[i] = W(i, 0);

		return model;
	}

	// LinearFitClassificationRosenblatt
	double* LinearFitClassificationClassic(double* model, double* inputs, int inputsSize, int inputSize, double* outputs, int outputSize)
	{
		int nbIter = 100;

		for (int k = 0; k < nbIter; k++)
		{
			for (int i = 0; i < inputsSize; i++)
			{
				double fitCalcul = model[0];
			
				for (int j = 0; j < inputSize; j++)
				{
					fitCalcul += model[j + 1] * inputs[i * inputSize + j];
				}

				if (fitCalcul != outputs[i])
				{
					int outPer = fitCalcul > 0 ? 1 : 0;
					model[0] += appro * (outputs[i] - outPer) * 1;

					for (int j = 0; j < inputSize; j++)
					{
						model[j + 1] += appro * (outputs[i] - outPer) *  inputs[i * inputSize + j];
					}
				}
			}
		}

		return model;
	}

	double* LinearFitClassificationHebb(double* model, double* inputs, int inputsSize, int inputSize, double* outputs, int outputSize)
	{
		return nullptr;
	}
	
	double* LinearClassify(double* model, double* inputs, int inputsSize)
	{
		double fitCalcul = model[0];

		for (int i = 0; i < inputsSize; i++)
		{
			fitCalcul += model[i + 1] * inputs[i];
		}

		double* output = new double[1];
		if (fitCalcul >= 0)
		{
			output[0] = 1;
		}
		else
		{
			output[0] = -1;
		}
		return output;
	}

	double LinearRegression(double* model, double* inputs, int inputsSize)
	{
		MatrixXd X(1, inputsSize + 1);
		MatrixXd Y(1, 1);
		MatrixXd W(inputsSize + 1, 1);

		X(0, 0) = 1;
		for (int i = 0; i < inputsSize; i++)
		{
			X(0, i + 1) = inputs[i];
		}

		for (int i = 0; i <= inputsSize; i++)
		{
			W(i, 0) = model[i];
		}

		Y = X * W;

		return Y(0,0);
	}

	double* LinearPredict(double* model, double* inputs, int inputsSize)
	{
		return nullptr;
	}



	double*** LinearCreateMLPModel(int nbCouches, int* nbNeurones, int nbInputs)
	{
		double*** model = new double**[nbCouches];
		for (int i = 0; i < nbCouches; i++)
		{
			int nbNeuronesInLayer = nbNeurones[i];
			model[i] = new double*[nbNeuronesInLayer];
			for (int j = 0; j < nbNeuronesInLayer; j++)
			{
				model[i][j] = new double[nbInputs + 1];
				for (int k = 0; k < nbInputs + 1; k++)
				{
					model[i][j][k] = rand() % 5 - 2;
				}
			}
		}

		return model;
	}


	double* LinearFitClassificationMulti(double*** model, int nbCouches, int* nbNeurones, double* inputs, int inputsSize, int inputSize, double* outputs, int outputSize, int nbIter)
	{
		float alpha = 0.1;
		for (int k = 0; k < nbIter; k++)
		{
			alpha = 0.1;
			for (int i = 0; i < inputsSize; i++)
			{
				double** outNeurones = new double*[nbCouches + 1];
				outNeurones[0] = new double[inputSize];

				for (int j = 0; j < inputSize; j++)
				{
					outNeurones[0][j] = inputs[i * inputSize + j];
				}

				for (int j = 0; j < nbCouches; j++)
				{
					int nbNeuronesInLayer = nbNeurones[j];
					outNeurones[j + 1] = new double[nbNeuronesInLayer];
				}

				for (int couche = 1; couche < nbCouches + 1; couche++)
				{
					for (int neurone = 0; neurone < nbNeurones[couche - 1]; neurone++)
					{
						double result = model[couche - 1][neurone][0] * 1;

						int nbPrevNeuronesInLayer = (couche - 1 == 0) ? 2 : nbNeurones[couche - 2];
						for (int previousNeurone = 0; previousNeurone < nbPrevNeuronesInLayer; previousNeurone++)
						{
							result += model[couche - 1][neurone][previousNeurone + 1] * outNeurones[couche - 1][previousNeurone];
						}
						outNeurones[couche][neurone] = tanh(result);
					}
				}

				double** gradientRetro = new double*[nbCouches];
				for (int couche = nbCouches - 1; couche >= 0; couche--)
				{
					gradientRetro[couche] = new double[nbNeurones[couche]];

					if (couche == nbCouches - 1)
					{
						for (int neurone = 0; neurone < nbNeurones[couche]; neurone++)
						{
							gradientRetro[couche][neurone] = (1 - pow(outNeurones[couche + 1][neurone], 2)) * (outNeurones[couche + 1][neurone] - outputs[i]);

							int nbPrevNeuronesInLayer = (couche - 1 == 0) ? 2 : nbNeurones[couche - 2];
							for (int previousNeurone = 0; previousNeurone < nbPrevNeuronesInLayer; previousNeurone++)
							{
								model[couche][neurone][previousNeurone] -= alpha * outNeurones[couche][neurone] * gradientRetro[couche][neurone];
							}
						}
					}
					else
					{
						for (int neurone = 0; neurone < nbNeurones[couche]; neurone++)
						{
							gradientRetro[couche][neurone] = (1 - pow(outNeurones[couche][neurone], 2));
							int somme = 0;

							int nbPrevNeuronesInLayer = (couche - 1 == 0) ? 2 : nbNeurones[couche - 2];

							for (int prevNeurone = 0; prevNeurone < nbNeurones[couche + 1]; prevNeurone++)
							{
								for (int poid = 0; poid < nbPrevNeuronesInLayer; poid++)
								{
									somme += model[couche + 1][prevNeurone][poid] * gradientRetro[couche + 1][prevNeurone];
								}
							}
							gradientRetro[couche][neurone] *= somme;

							for (int previousNeurone = 0; previousNeurone < nbPrevNeuronesInLayer; previousNeurone++)
							{
								model[couche][neurone][previousNeurone] -= alpha * outNeurones[couche][neurone] * gradientRetro[couche][neurone];
							}
						}
					}
				}
			}
		}
		return nullptr;
	}

	double MultiClassify(double*** model, double* inputs, int nbCouches, int* nbNeurones, int inputsSize)
	{
		double** outNeurones = new double*[nbCouches + 1];
		outNeurones[0] = new double[inputsSize];

		for (int j = 0; j < inputsSize; j++)
		{
			outNeurones[0][j] = inputs[j];
		}

		for (int j = 0; j < nbCouches; j++)
		{
			int nbNeuronesInLayer = nbNeurones[j];
			outNeurones[j + 1] = new double[nbNeuronesInLayer];
		}

		for (int couche = 0; couche < nbCouches; couche++)
		{
			for (int neurone = 0; neurone < nbNeurones[couche]; neurone++)
			{
				double fitCalcul = model[couche][neurone][0] * 1;
				int nbPoids = (couche == 0) ? 2 : nbNeurones[couche - 1];
				for (int poid = 1; poid <= nbPoids; poid++)
				{
					fitCalcul += model[couche][neurone][poid] * outNeurones[couche - 1][poid];
				}
				outNeurones[couche][neurone] = tanh(fitCalcul);
			}
		}

		return outNeurones[nbCouches][0];
	}
}