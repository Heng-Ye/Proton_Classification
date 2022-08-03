#include <stdio.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace ROOT::Math;

void root2csv_converter() {
	//setup input file name ---------------------//
	TString str_file=Form("protons_mva2.root");

	//setup output file name -----------------------//
	TString str_out_file=Form("protons_mva2.csv");
	ofstream fout;
  	fout.open(str_out_file.Data());
	
	//get tree ----------------------------------//
	TFile *file = TFile::Open(str_file.Data());
	TTree *tr = (TTree *)file->Get("tr");

	//define variables -------------------//
	Bool_t train; //train sample or not
	Int_t tag;
	Float_t ntrklen;
	Float_t trklen;
	Float_t PID;
	Float_t B;
	Float_t costheta;
	Float_t mediandedx;
	Float_t endpointdedx;
	Float_t calo;
	Float_t avcalo;

 	//set branch address ---------------------------------//
	tr->SetBranchAddress("train",&train);
	tr->SetBranchAddress("tag",&tag);
	tr->SetBranchAddress("ntrklen",&ntrklen);
	tr->SetBranchAddress("trklen",&trklen);
	tr->SetBranchAddress("PID",&PID);
	tr->SetBranchAddress("B",&B);
	tr->SetBranchAddress("costheta",&costheta);
	tr->SetBranchAddress("mediandedx",&mediandedx);
	tr->SetBranchAddress("endpointdedx",&endpointdedx);
	tr->SetBranchAddress("calo",&calo);
	tr->SetBranchAddress("avcalo",&avcalo);

	//save header to the csv file --------------------------------------------------------//
  	fout<<"train,tag,ntrklen,trklen,PID,B,costheta,mediandedx,endpointdedx,calo,avcalo\n";

	//loop over all events ------------------------------------//
	Long64_t nentries = tr->GetEntries();
	std::cout<<"nentries: "<<nentries<<std::endl;

	for(int i=0; i<tr->GetEntries(); i++){
	//for(int i=0; i<20; i++){
	   tr->GetEntry(i);
           if (i%1000==0) std::cout<<i<<"/"<<nentries<<std::endl;
	   //if (i<10) cout<<"trklen: "<<trklen<<endl;
	   fout<<train<<","<<tag<<","<<ntrklen<<","<<trklen<<","<<PID<<","<<B<<","<<costheta<<","<<mediandedx<<","<<endpointdedx<<","<<calo<<","<<avcalo<<"\n";	

	}
  	fout.close();
	   

}
