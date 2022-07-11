#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"

void TMVAClassification(TString myMethodList, TString inputFileName, TString outputFileName) {
	//Choose the classification methods you want 
	//TString myMethodList = "CutsPCA,LikelihoodPCA,LikelihoodKDE,LikelihoodMIX,"; //"method1, method2"

	//TString inputFileName = "./protons.root";
	//TString outputFileName="proton_TMVA_ClassificationOutput.root";

	// Default MVA methods to be trained + tested
	std::map<std::string,int> Use;

	//---------------------------------------------------------------
	// Cut optimisation
	Use["Cuts"]            = 1;
	Use["CutsD"]           = 1;
	Use["CutsPCA"]         = 0;
	Use["CutsGA"]          = 0;
	Use["CutsSA"]          = 0;
	//
	// 1-dimensional likelihood ("naive Bayes estimator")
	Use["Likelihood"]      = 1;
	Use["LikelihoodD"]     = 0; // the "D" extension indicates decorrelated input variables (see option strings)
	Use["LikelihoodPCA"]   = 1; // the "PCA" extension indicates PCA-transformed input variables (see option strings)
	Use["LikelihoodKDE"]   = 0;
	Use["LikelihoodMIX"]   = 0;
	//
	// Mutidimensional likelihood and Nearest-Neighbour methods
	Use["PDERS"]           = 1;
	Use["PDERSD"]          = 0;
	Use["PDERSPCA"]        = 0;
	Use["PDEFoam"]         = 1;
	Use["PDEFoamBoost"]    = 0; // uses generalised MVA method boosting
	Use["KNN"]             = 1; // k-nearest neighbour method
	//
	// Linear Discriminant Analysis
	Use["LD"]              = 1; // Linear Discriminant identical to Fisher
	Use["Fisher"]          = 0;
	Use["FisherG"]         = 0;
	Use["BoostedFisher"]   = 0; // uses generalised MVA method boosting
	Use["HMatrix"]         = 0;
	//
	// Function Discriminant analysis
	Use["FDA_GA"]          = 1; // minimisation of user-defined function using Genetics Algorithm
	Use["FDA_SA"]          = 0;
	Use["FDA_MC"]          = 0;
	Use["FDA_MT"]          = 0;
	Use["FDA_GAMT"]        = 0;
	Use["FDA_MCMT"]        = 0;
	//
	// Neural Networks (all are feed-forward Multilayer Perceptrons)
	Use["MLP"]             = 0; // Recommended ANN
	Use["MLPBFGS"]         = 0; // Recommended ANN with optional training method
	Use["MLPBNN"]          = 1; // Recommended ANN with BFGS training method and bayesian regulator
	Use["CFMlpANN"]        = 0; // Depreciated ANN from ALEPH
	Use["TMlpANN"]         = 0; // ROOT's own ANN
#ifdef R__HAS_TMVAGPU
	Use["DNN_GPU"]         = 1; // CUDA-accelerated DNN training.
#else
	Use["DNN_GPU"]         = 0;
#endif

#ifdef R__HAS_TMVACPU
	Use["DNN_CPU"]         = 1; // Multi-core accelerated DNN.
#else
	Use["DNN_CPU"]         = 0;
#endif
	//
	// Support Vector Machine
	Use["SVM"]             = 1;
	//
	// Boosted Decision Trees
	Use["BDT"]             = 1; // uses Adaptive Boost
	Use["BDTG"]            = 0; // uses Gradient Boost
	Use["BDTB"]            = 0; // uses Bagging
	Use["BDTD"]            = 0; // decorrelation + Adaptive Boost
	Use["BDTF"]            = 0; // allow usage of fisher discriminant for node splitting
	//
	// Friedman's RuleFit method, ie, an optimised series of cuts ("rules")
	Use["RuleFit"]         = 1;
	// ---------------------------------------------------------------

	std::cout << std::endl;
	std::cout << "==> Start TMVAClassification" << std::endl;

	// Select methods (don't look at this code - not of interest)
	if (myMethodList != "") {
		for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) it->second = 0;

		std::vector<TString> mlist = TMVA::gTools().SplitString( myMethodList, ',' );
		for (UInt_t i=0; i<mlist.size(); i++) {
			std::string regMethod(mlist[i]);

			if (Use.find(regMethod) == Use.end()) {
				std::cout << "Method \"" << regMethod << "\" not known in TMVA under this name. Choose among the following:" << std::endl;
				for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) std::cout << it->first << " ";
				std::cout << std::endl;
				return 1;
			}
			Use[regMethod] = 1;
		}
	}

	TMVA::Tools::Instance(); //loads the library
	std::cout << "==> Start TMVAClassification" << std::endl;

	//I/O
	auto inputFile = TFile::Open(inputFileName.Data());
	auto outputFile = TFile::Open(outputFileName.Data(), "RECREATE");
	TMVA::Factory *factory = new TMVA::Factory("TMVAClassification", outputFile, "!V:ROC:!Silent:Color:!DrawProgressBar:AnalysisType=Classification" );


	//register the training and test trees
	TTree *signalTree     = (TTree*)inputFile->Get("tr"); //HY::Signal and bkg tree are the same but using different true labels for training
	TTree *backgroundTree = (TTree*)inputFile->Get("tr");

	TMVA::DataLoader * loader = new TMVA::DataLoader("dataset");

	// global event weights per tree (see below for setting event-wise weights)
	Double_t signalWeight     = 1.0;
	Double_t backgroundWeight = 1.0;

	// You can add an arbitrary number of signal or background trees
	loader->AddSignalTree    ( signalTree,     signalWeight     );
	loader->AddBackgroundTree( backgroundTree, backgroundWeight );

	signalTree->Print();

	//Add variables for training
	//loader->AddVariable( "train", 'O' ); //train or test set
	//loader->AddVariable( "tag", 'I' ); //true labels
	loader->AddVariable( "ntrklen", 'D' ); //normalized track length
	loader->AddVariable( "trklen", 'D' ); //track length
	loader->AddVariable( "PID", 'D' ); //chi^2-pid (stopping proton hypothesis)
	loader->AddVariable( "B", 'D' ); //impact parameter
	loader->AddVariable( "mediandedx", 'D' ); //median dedx
	loader->AddVariable( "endpointdedx", 'D' ); //endpoint dedx
	loader->AddVariable( "calo", 'D' ); //calorimetric energy
	loader->AddVariable( "avcalo", 'D' ); //calorimetric energy/trklen
	loader->AddVariable( "costheta", 'D' ); //costheta

	//true label list
	//tag=1; //kinel
	//tag=2; //kel
	//tag=3; //kMIDcosmic
	//tag=4; //kMIDpi
	//tag=5; //kMIDp
	//tag=6; //kMIDmu
	//tag=7; //kMIDeg
	//tag=8; //kMIDother

	// You can add so-called "Spectator variables", which are not used in the MVA training,
	// but will appear in the final "TestTree" produced by TMVA. This TestTree will contain the
	// input variables, the response values of all trained MVAs, and the spectator variables
	//loader->AddSpectator( "spec1 := var1*2",  "Spectator 1", "units", 'F' );
	//loader->AddSpectator( "spec2 := var1*3",  "Spectator 2", "units", 'F' );

	//  We can define also the event weights
	// Set individual event weights (the variables must exist in the original TTree)
	//    for signal    : factory->SetSignalWeightExpression    ("weight1*weight2");
	//    for background: factory->SetBackgroundWeightExpression("weight1*weight2");
	//loader->SetBackgroundWeightExpression( "weight" );


	// Apply additional cuts on the signal and background samples (can be different)
	TCut mycuts = "tag==1"; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
	TCut mycutb = "tag>1"; // for example: TCut mycutb = "abs(var1)<0.5";


	// Tell the factory how to use the training and testing events
	// If no numbers of events are given, half of the events in the tree are used for training, and the other half for testing:
	//    loader->PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );
	// To also specify the number of testing events, use:
	//    loader->PrepareTrainingAndTestTree( mycut,
	//                                         "NSigTrain=3000:NBkgTrain=3000:NSigTest=3000:NBkgTest=3000:SplitMode=Random:!V" );
	//loader->PrepareTrainingAndTestTree( mycuts, mycutb, "nTrain_Signal=4000:nTrain_Background=2000:SplitMode=Random:NormMode=NumEvents:!V" );
	loader->PrepareTrainingAndTestTree(mycuts, mycutb, "SplitMode=random:!V" );



	// ### Book MVA methods
	//
	// Please lookup the various method configuration options in the corresponding cxx files, eg:
	// src/MethoCuts.cxx, etc, or here: http://tmva.sourceforge.net/optionRef.html
	// it is possible to preset ranges in the option string in which the cut optimisation should be done:
	// "...:CutRangeMin[2]=-1:CutRangeMax[2]=1"...", where [2] is the third input variable

	// Cut optimisation
	if (Use["Cuts"])
		factory->BookMethod( loader, TMVA::Types::kCuts, "Cuts",
				"!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart" );

	if (Use["CutsD"])
		factory->BookMethod( loader, TMVA::Types::kCuts, "CutsD",
				"!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart:VarTransform=Decorrelate" );

	if (Use["CutsPCA"])
		factory->BookMethod( loader, TMVA::Types::kCuts, "CutsPCA",
				"!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart:VarTransform=PCA" );

	if (Use["CutsGA"])
		factory->BookMethod( loader, TMVA::Types::kCuts, "CutsGA",
				"H:!V:FitMethod=GA:CutRangeMin[0]=-10:CutRangeMax[0]=10:VarProp[1]=FMax:EffSel:Steps=30:Cycles=3:PopSize=400:SC_steps=10:SC_rate=5:SC_factor=0.95" );

	if (Use["CutsSA"])
		factory->BookMethod( loader, TMVA::Types::kCuts, "CutsSA",
				"!H:!V:FitMethod=SA:EffSel:MaxCalls=150000:KernelTemp=IncAdaptive:InitialTemp=1e+6:MinTemp=1e-6:Eps=1e-10:UseDefaultScale" );

	// Likelihood ("naive Bayes estimator")
	if (Use["Likelihood"])
		factory->BookMethod( loader, TMVA::Types::kLikelihood, "Likelihood",
				"H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmoothBkg[1]=10:NSmooth=1:NAvEvtPerBin=50" );

	// Decorrelated likelihood
	if (Use["LikelihoodD"])
		factory->BookMethod( loader, TMVA::Types::kLikelihood, "LikelihoodD",
				"!H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmooth=5:NAvEvtPerBin=50:VarTransform=Decorrelate" );

	// PCA-transformed likelihood
	if (Use["LikelihoodPCA"])
		factory->BookMethod( loader, TMVA::Types::kLikelihood, "LikelihoodPCA",
				"!H:!V:!TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmooth=5:NAvEvtPerBin=50:VarTransform=PCA" );

	// Use a kernel density estimator to approximate the PDFs
	if (Use["LikelihoodKDE"])
		factory->BookMethod( loader, TMVA::Types::kLikelihood, "LikelihoodKDE",
				"!H:!V:!TransformOutput:PDFInterpol=KDE:KDEtype=Gauss:KDEiter=Adaptive:KDEFineFactor=0.3:KDEborder=None:NAvEvtPerBin=50" );

	// Use a variable-dependent mix of splines and kernel density estimator
	if (Use["LikelihoodMIX"])
		factory->BookMethod( loader, TMVA::Types::kLikelihood, "LikelihoodMIX",
				"!H:!V:!TransformOutput:PDFInterpolSig[0]=KDE:PDFInterpolBkg[0]=KDE:PDFInterpolSig[1]=KDE:PDFInterpolBkg[1]=KDE:PDFInterpolSig[2]=Spline2:PDFInterpolBkg[2]=Spline2:PDFInterpolSig[3]=Spline2:PDFInterpolBkg[3]=Spline2:KDEtype=Gauss:KDEiter=Nonadaptive:KDEborder=None:NAvEvtPerBin=50" );

	// Test the multi-dimensional probability density estimator
	// here are the options strings for the MinMax and RMS methods, respectively:
	//
	//      "!H:!V:VolumeRangeMode=MinMax:DeltaFrac=0.2:KernelEstimator=Gauss:GaussSigma=0.3" );
	//      "!H:!V:VolumeRangeMode=RMS:DeltaFrac=3:KernelEstimator=Gauss:GaussSigma=0.3" );
	if (Use["PDERS"])
		factory->BookMethod( loader, TMVA::Types::kPDERS, "PDERS",
				"!H:!V:NormTree=T:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600" );

	if (Use["PDERSD"])
		factory->BookMethod( loader, TMVA::Types::kPDERS, "PDERSD",
				"!H:!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:VarTransform=Decorrelate" );

	if (Use["PDERSPCA"])
		factory->BookMethod( loader, TMVA::Types::kPDERS, "PDERSPCA",
				"!H:!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:VarTransform=PCA" );

	// Multi-dimensional likelihood estimator using self-adapting phase-space binning
	if (Use["PDEFoam"])
		factory->BookMethod( loader, TMVA::Types::kPDEFoam, "PDEFoam",
				"!H:!V:SigBgSeparate=F:TailCut=0.001:VolFrac=0.0666:nActiveCells=500:nSampl=2000:nBin=5:Nmin=100:Kernel=None:Compress=T" );

	if (Use["PDEFoamBoost"])
		factory->BookMethod( loader, TMVA::Types::kPDEFoam, "PDEFoamBoost",
				"!H:!V:Boost_Num=30:Boost_Transform=linear:SigBgSeparate=F:MaxDepth=4:UseYesNoCell=T:DTLogic=MisClassificationError:FillFoamWithOrigWeights=F:TailCut=0:nActiveCells=500:nBin=20:Nmin=400:Kernel=None:Compress=T" );

	// K-Nearest Neighbour classifier (KNN)
	if (Use["KNN"])
		factory->BookMethod( loader, TMVA::Types::kKNN, "KNN",
				"H:nkNN=20:ScaleFrac=0.8:SigmaFact=1.0:Kernel=Gaus:UseKernel=F:UseWeight=T:!Trim" ); //<--By default
		//factory->BookMethod( loader, TMVA::Types::kKNN, "KNN",
				//"H:nkNN=3:ScaleFrac=0.8:SigmaFact=1.0:Kernel=Gaus:UseKernel=F:UseWeight=T:!Trim" ); //HY:: set to 3 clusters

	// H-Matrix (chi2-squared) method
	if (Use["HMatrix"])
		factory->BookMethod( loader, TMVA::Types::kHMatrix, "HMatrix", "!H:!V:VarTransform=None" );

	// Linear discriminant (same as Fisher discriminant)
	if (Use["LD"])
		factory->BookMethod( loader, TMVA::Types::kLD, "LD", "H:!V:VarTransform=None:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10" );

	// Fisher discriminant (same as LD)
	if (Use["Fisher"])
		factory->BookMethod( loader, TMVA::Types::kFisher, "Fisher", "H:!V:Fisher:VarTransform=None:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10" );

	// Fisher with Gauss-transformed input variables
	if (Use["FisherG"])
		factory->BookMethod( loader, TMVA::Types::kFisher, "FisherG", "H:!V:VarTransform=Gauss" );

	// Composite classifier: ensemble (tree) of boosted Fisher classifiers
	if (Use["BoostedFisher"])
		factory->BookMethod( loader, TMVA::Types::kFisher, "BoostedFisher",
				"H:!V:Boost_Num=20:Boost_Transform=log:Boost_Type=AdaBoost:Boost_AdaBoostBeta=0.2:!Boost_DetailedMonitoring" );

	// Function discrimination analysis (FDA) -- test of various fitters - the recommended one is Minuit (or GA or SA)
	if (Use["FDA_MC"])
		factory->BookMethod( loader, TMVA::Types::kFDA, "FDA_MC",
				"H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MC:SampleSize=100000:Sigma=0.1" );

	if (Use["FDA_GA"]) // can also use Simulated Annealing (SA) algorithm (see Cuts_SA options])
		factory->BookMethod( loader, TMVA::Types::kFDA, "FDA_GA",
				"H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=GA:PopSize=100:Cycles=2:Steps=5:Trim=True:SaveBestGen=1" );

	if (Use["FDA_SA"]) // can also use Simulated Annealing (SA) algorithm (see Cuts_SA options])
		factory->BookMethod( loader, TMVA::Types::kFDA, "FDA_SA",
				"H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=SA:MaxCalls=15000:KernelTemp=IncAdaptive:InitialTemp=1e+6:MinTemp=1e-6:Eps=1e-10:UseDefaultScale" );

	if (Use["FDA_MT"])
		factory->BookMethod( loader, TMVA::Types::kFDA, "FDA_MT",
				"H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=2:UseImprove:UseMinos:SetBatch" );

	if (Use["FDA_GAMT"])
		factory->BookMethod( loader, TMVA::Types::kFDA, "FDA_GAMT",
				"H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=GA:Converger=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=0:!UseImprove:!UseMinos:SetBatch:Cycles=1:PopSize=5:Steps=5:Trim" );

	if (Use["FDA_MCMT"])
		factory->BookMethod( loader, TMVA::Types::kFDA, "FDA_MCMT",
				"H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MC:Converger=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=0:!UseImprove:!UseMinos:SetBatch:SampleSize=20" );

	// TMVA ANN: MLP (recommended ANN) -- all ANNs in TMVA are Multilayer Perceptrons
	if (Use["MLP"])
		factory->BookMethod( loader, TMVA::Types::kMLP, "MLP", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:!UseRegulator" );

	if (Use["MLPBFGS"])
		factory->BookMethod( loader, TMVA::Types::kMLP, "MLPBFGS", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:TrainingMethod=BFGS:!UseRegulator" );

	if (Use["MLPBNN"])
		factory->BookMethod( loader, TMVA::Types::kMLP, "MLPBNN", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=60:HiddenLayers=N+5:TestRate=5:TrainingMethod=BFGS:UseRegulator" ); // BFGS training with bayesian regulators


	// Multi-architecture DNN implementation.
	if (Use["DNN_CPU"] or Use["DNN_GPU"]) {
		// General layout.
		TString layoutString ("Layout=TANH|128,TANH|128,TANH|128,LINEAR");

		// Training strategies.
		TString training0("LearningRate=1e-2,Momentum=0.9,Repetitions=1,"
				"ConvergenceSteps=30,BatchSize=256,TestRepetitions=10,"
				"WeightDecay=1e-4,Regularization=None,"
				"DropConfig=0.0+0.5+0.5+0.5, Multithreading=True");
		TString training1("LearningRate=1e-2,Momentum=0.9,Repetitions=1,"
				"ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
				"WeightDecay=1e-4,Regularization=L2,"
				"DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
		TString training2("LearningRate=1e-3,Momentum=0.0,Repetitions=1,"
				"ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
				"WeightDecay=1e-4,Regularization=L2,"
				"DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
		TString trainingStrategyString ("TrainingStrategy=");
		trainingStrategyString += training0 + "|" + training1 + "|" + training2;

		// General Options.
		TString dnnOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:"
				"WeightInitialization=XAVIERUNIFORM");
		dnnOptions.Append (":"); dnnOptions.Append (layoutString);
		dnnOptions.Append (":"); dnnOptions.Append (trainingStrategyString);

		// Cuda implementation.
		if (Use["DNN_GPU"]) {
			TString gpuOptions = dnnOptions + ":Architecture=GPU";
			factory->BookMethod(loader, TMVA::Types::kDL, "DNN_GPU", gpuOptions);
		}
		// Multi-core CPU implementation.
		if (Use["DNN_CPU"]) {
			TString cpuOptions = dnnOptions + ":Architecture=CPU";
			factory->BookMethod(loader, TMVA::Types::kDL, "DNN_CPU", cpuOptions);
		}
	}

	// CF(Clermont-Ferrand)ANN
	if (Use["CFMlpANN"])
		factory->BookMethod( loader, TMVA::Types::kCFMlpANN, "CFMlpANN", "!H:!V:NCycles=200:HiddenLayers=N+1,N"  ); // n_cycles:#nodes:#nodes:...

	// Tmlp(Root)ANN
	if (Use["TMlpANN"])
		factory->BookMethod( loader, TMVA::Types::kTMlpANN, "TMlpANN", "!H:!V:NCycles=200:HiddenLayers=N+1,N:LearningMethod=BFGS:ValidationFraction=0.3"  ); // n_cycles:#nodes:#nodes:...

	// Support Vector Machine
	if (Use["SVM"])
		factory->BookMethod( loader, TMVA::Types::kSVM, "SVM", "Gamma=0.25:Tol=0.001:VarTransform=Norm" );

	// Boosted Decision Trees
	if (Use["BDTG"]) // Gradient Boost
		factory->BookMethod( loader, TMVA::Types::kBDT, "BDTG",
				"!H:!V:NTrees=1000:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=2" );

	if (Use["BDT"])  // Adaptive Boost
		factory->BookMethod( loader, TMVA::Types::kBDT, "BDT",
				"!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );

	if (Use["BDTB"]) // Bagging
		factory->BookMethod( loader, TMVA::Types::kBDT, "BDTB",
				"!H:!V:NTrees=400:BoostType=Bagging:SeparationType=GiniIndex:nCuts=20" );

	if (Use["BDTD"]) // Decorrelation + Adaptive Boost
		factory->BookMethod( loader, TMVA::Types::kBDT, "BDTD",
				"!H:!V:NTrees=400:MinNodeSize=5%:MaxDepth=3:BoostType=AdaBoost:SeparationType=GiniIndex:nCuts=20:VarTransform=Decorrelate" );

	if (Use["BDTF"])  // Allow Using Fisher discriminant in node splitting for (strong) linearly correlated variables
		factory->BookMethod( loader, TMVA::Types::kBDT, "BDTF",
				"!H:!V:NTrees=50:MinNodeSize=2.5%:UseFisherCuts:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20" );

	// RuleFit -- TMVA implementation of Friedman's method
	if (Use["RuleFit"])
		factory->BookMethod( loader, TMVA::Types::kRuleFit, "RuleFit",
				"H:!V:RuleFitModule=RFTMVA:Model=ModRuleLinear:MinImp=0.001:RuleMinDist=0.001:NTrees=20:fEventsMin=0.01:fEventsMax=0.5:GDTau=-1.0:GDTauPrec=0.01:GDStep=0.01:GDNSteps=10000:GDErrScale=1.02" );





	//Book Methods
	// Likelihood ("naive Bayes estimator")
	//factory.BookMethod(loader, TMVA::Types::kLikelihood, "Likelihood",
	//		"H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmoothBkg[1]=10:NSmooth=1:NAvEvtPerBin=50" );

	// Use a kernel density estimator to approximate the PDFs
	//factory.BookMethod(loader, TMVA::Types::kLikelihood, "LikelihoodKDE",
	//		"!H:!V:!TransformOutput:VarTransform=D:PDFInterpol=KDE:KDEtype=Gauss:KDEiter=Adaptive:KDEFineFactor=0.3:KDEborder=None:NAvEvtPerBin=50" ); 


	// Fisher discriminant (same as LD)
	//factory.BookMethod(loader, TMVA::Types::kFisher, "Fisher", "H:!V:Fisher:VarTransform=None:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10" );

	//Boosted Decision Trees
	//factory.BookMethod(loader,TMVA::Types::kBDT, "BDT",
	//		"!V:NTrees=200:MinNodeSize=2.5%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );

	//Multi-Layer Perceptron (Neural Network)
	//factory.BookMethod(loader, TMVA::Types::kMLP, "MLP",
	//		"!H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+5:TestRate=5:!UseRegulator" );














	/*
	   std::map<std::string,int> Use; //efault MVA methods to be trained + tested

	//Classification Methods ------------------------------------------//
	TString myMethodList = ""; //classification methods to be used
	//"myMethod1,myMethod2,myMethod3"	
	//Method lists:

	// Cut optimisation
	//Use["Cuts"]            = 1;
	//Use["CutsD"]           = 1;
	//Use["CutsPCA"]         = 0;
	//Use["CutsGA"]          = 0;
	//Use["CutsSA"]          = 0;	

	// 1-dimensional likelihood ("naive Bayes estimator")
	//Use["Likelihood"]      = 1;
	//Use["LikelihoodD"]     = 0; // the "D" extension indicates decorrelated input variables (see option strings)
	//Use["LikelihoodPCA"]   = 1; // the "PCA" extension indicates PCA-transformed input variables (see option strings)
	//Use["LikelihoodKDE"]   = 0;
	//Use["LikelihoodMIX"]   = 0;

	// Mutidimensional likelihood and Nearest-Neighbour methods
	//Use["PDERS"]           = 1;
	//Use["PDERSD"]          = 0;
	//Use["PDERSPCA"]        = 0;
	//Use["PDEFoam"]         = 1;
	//Use["PDEFoamBoost"]    = 0; // uses generalised MVA method boosting
	//Use["KNN"]             = 1; // k-nearest neighbour method

	// Linear Discriminant Analysis
	//Use["LD"]              = 1; // Linear Discriminant identical to Fisher
	//Use["Fisher"]          = 0;
	//Use["FisherG"]         = 0;
	//Use["BoostedFisher"]   = 0; // uses generalised MVA method boosting
	//Use["HMatrix"]         = 0;

	// Function Discriminant analysis
	//Use["FDA_GA"]          = 1; // minimisation of user-defined function using Genetics Algorithm
	//Use["FDA_SA"]          = 0;
	//Use["FDA_MC"]          = 0;
	//Use["FDA_MT"]          = 0;
	//Use["FDA_GAMT"]        = 0;
	//Use["FDA_MCMT"]        = 0;

	// Neural Networks (all are feed-forward Multilayer Perceptrons)
	//Use["MLP"]             = 0; // Recommended ANN
	//Use["MLPBFGS"]         = 0; // Recommended ANN with optional training method
	//Use["MLPBNN"]          = 1; // Recommended ANN with BFGS training method and bayesian regulator
	//Use["CFMlpANN"]        = 0; // Depreciated ANN from ALEPH
	//Use["TMlpANN"]         = 0; // ROOT's own ANN
	//#ifdef R__HAS_TMVAGPU
	//Use["DNN_GPU"]         = 1; // CUDA-accelerated DNN training.
	//#else
	//Use["DNN_GPU"]         = 0;
	//#endif

	//#ifdef R__HAS_TMVACPU
	//Use["DNN_CPU"]         = 1; // Multi-core accelerated DNN.
	//#else
	//Use["DNN_CPU"]         = 0;
	//#endif
	//
	// Support Vector Machine
	//Use["SVM"]             = 1;
	//
	// Boosted Decision Trees
	//Use["BDT"]             = 1; // uses Adaptive Boost
	//Use["BDTG"]            = 0; // uses Gradient Boost
	//Use["BDTB"]            = 0; // uses Bagging
	//Use["BDTD"]            = 0; // decorrelation + Adaptive Boost
	//Use["BDTF"]            = 0; // allow usage of fisher discriminant for node splitting
	//
	// Friedman's RuleFit method, ie, an optimised series of cuts ("rules")
	//Use["RuleFit"]         = 1;
	//-----------------------------------------------------------------//
	*/




	factory->TrainAllMethods();
	factory->TestAllMethods();  
	factory->EvaluateAllMethods();  

	//We enable JavaScript visualisation for the plots
	//%jsroot on
	//
	//auto c1 = factory.GetROCCurve(loader);
	//c1->Draw();

	outputFile->Close();


}

