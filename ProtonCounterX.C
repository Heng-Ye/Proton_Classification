#define ProtonCounter_cxx
#include "ProtonCounter.h"

#include <TH2.h>
#include <TH1.h>
#include "TH2D.h"
#include <TStyle.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TLegendEntry.h>

#include <TMath.h>
#include <TLine.h>
#include <TF1.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TH3F.h>
#include <TString.h>
#include <TProfile2D.h>
#include <THStack.h>
#include "TGraph.h" 
#include "TGraphSmooth.h" 
#include "TParameter.h"
#include "TGraphErrors.h"
#include "string"
#include "vector"
#include "TSpline.h"
#include "TH3F.h"
#include <TMath.h>
#include <TGraph2D.h>
#include <TRandom2.h>
#include <Math/Functor.h>
#include <TPolyLine3D.h>
#include <Math/Vector3D.h>
#include <Fit/Fitter.h>
#include "TVector3.h"

#include <stdio.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace std;
using namespace ROOT::Math;

void ProtonCounter::Loop() {
	if (fChain == 0) return;

	Int_t n_inel=0; //pure inel events

	//Name of output file ------------------------------------------------------------------------------------------------------------//
	Long64_t nentries = fChain->GetEntries();
	std::cout<<"nentries: "<<nentries<<std::endl;

	Long64_t nbytes = 0, nb = 0;
	bool isTestSample=true;

	for (Long64_t jentry=0; jentry<nentries;jentry++) { //main entry loop
		//for (Long64_t jentry=0; jentry<4000;jentry++) { //main entry loop
		Long64_t ientry = LoadTree(jentry);
		if (ientry < 0) break;
		nb = fChain->GetEntry(jentry);   nbytes += nb;

		isTestSample = true;
		if (ientry%2 == 0) { 
			isTestSample = false; //Divide MC sample by 2 parts: test+ufold
			train=0;
		}
		//if (isTestSample) continue; //only validate sample
		//if (!isTestSample) continue; //only test sample

		//true_sliceID = -1;
		//reco_sliceID = -1;

		//only select protons	
		//if (primary_truth_Pdg!=pdg) continue; //only interested in protons
		if (beamtrackPdg!=pdg) continue; //only interested in protons
		if (jentry%1000==0) std::cout<<jentry<<"/"<<nentries<<std::endl;

		//std::cout<<"beamtrackPdg:"<<beamtrackPdg<<std::endl;
		//n_tot++;

		//Event Selection Cut -- Part 1 ----------------------------------//
		bool IsBeamMatch=false; //if recostructed the right track (recoID=truthID)
		bool IsPandoraSlice=false; //pandora slice cut (can pandora reconstruct this track)
		bool IsCaloSize=false; //if calo size not empty
		bool IsIntersection=false; //if any track intersect with our reco track		
		if (primary_truth_Isbeammatched==1) IsBeamMatch=true;
		if (isprimarytrack==1&&isprimaryshower==0) IsPandoraSlice=true; 
		if (!primtrk_hitz->empty()) IsCaloSize=true;
		if (timeintersection->size()) IsIntersection=true;
		//----------------------------------------------------------------//

		//Truth label of Primarytrack_End ------------------------------------------------------------------------------------------------//
		bool IsPureInEL=false; //inel
		bool IsPureEL=false; //el
		//bool IsPureMCS=false; //no hadron scattering

		//if (primary_truth_EndProcess->c_str()!=NULL) n_true_end++;
		if (strcmp(primary_truth_EndProcess->c_str(),"protonInelastic")==0) {
			IsPureInEL=true;
			n_inel++;
		}
		else { //hIoni
			IsPureEL=true;
			/*
			   if (interactionProcesslist->size()) { //size of interactionProcesslist >=0
			   cout<<"interactionProcesslist->size():"<<interactionProcesslist->size()<<endl;	
			   for(size_t iiii=0; iiii<interactionProcesslist->size(); iiii++) { //loop over all true interaction hits in this track
			   try {
			   double intx=interactionX->at(iiii);
			   double inty=interactionY->at(iiii);
			   double intz=interactionZ->at(iiii); 
			   cout<<"["<<iiii<<"] process:"<<interactionProcesslist->at(iiii)<<" z:"<<intz<<endl;

			   if(strcmp(interactionProcesslist->at(iiii).c_str(),"hadElastic")==0) {
			   IsPureEL=1;
			   }
			   }
			   catch (const std::out_of_range & ex) {
			   std::cout << "out_of_range Exception Caught :: interactionProcesslist" << ex.what() << std::endl;
			   n_processmap_error++;
			   }
			//if (intz<0) { //if interaction outside tpc
			//if(strcmp(interactionProcesslist->at(iiii).c_str(),"Transportation")!=0) {
			} //loop over all true interaction hits in this track 
			} //size of interactionProcesslist >=0
			*/
		} //hIoni

		//Get true start/end point -----------------------------------------------------------------------//
		double true_endz=primary_truth_EndPosition_MC[2]; 
		double true_endy=primary_truth_EndPosition_MC[1]; 
		double true_endx=primary_truth_EndPosition_MC[0];

		double true_stz=primary_truth_StartPosition_MC[2];
		double true_sty=primary_truth_StartPosition_MC[1];
		double true_stx=primary_truth_StartPosition_MC[0];

		bool IsTrueEndOutside=false;
		if (true_endz<0.) {
			IsTrueEndOutside=true;
		}

		//Evt Classification =====================================================================//
		//signal -----------------------------//
		bool kinel=false;
		bool kel=false;
		//bool kmcs=false;
		if (IsBeamMatch) { //beam-match
			if (IsPureInEL) { 
				kinel=true;
			}
			if (IsPureEL) { 
				kel=true;
			}
			//if (IsPureMCS) kmcs=true;
		} //beam-match

		//background ------------------------------------------------------------------------//
		bool kMIDcosmic=false; //beam or cosmic
		bool kMIDpi=false; //+-pi
		bool kMIDp=false; //p
		bool kMIDmu=false; //mu
		bool kMIDeg=false; //e/gamma
		bool kMIDother=false; //other
		if (!IsBeamMatch) { //!beam-match
			if (primary_truth_byE_origin==2) { 
				kMIDcosmic=true;
			}
			else if (std::abs(primary_truth_byE_PDG)==211) {
				kMIDpi=true;
			}
			else if (primary_truth_byE_PDG==2212) {
				kMIDp=true;
			}
			else if (std::abs(primary_truth_byE_PDG)==13) {
				kMIDmu=true;
			}
			else if (std::abs(primary_truth_byE_PDG)==11 || primary_truth_byE_PDG==22) {
				kMIDeg=true;
			}
			else {
				kMIDother=true;
			}
		} //!beam-match	





		} //main entry loop


		//print out results ----------------------------------------//
		std::cout<<"================================="<<std::endl;
		std::cout<<"MC total inel evts :"<<n_inel<<etd::endl;
		std::cout<<"=================================\n"<<std::endl;
		//----------------------------------------------------------//


		}
