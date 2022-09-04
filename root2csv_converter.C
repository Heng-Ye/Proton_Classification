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
	//setup input file name -----------------------------------//
	//TString str_filename="protons_with_additionalinfo_mva2";
	TString str_filename="./data_prep/protons_mva2_20220902";
	//TString str_file=Form("protons_mva2.root");
	TString str_file=Form("%s.root",str_filename.Data());

	//setup output file name ---------------------------------//
	//TString str_out_file=Form("protons_mva2.csv");
	//TString str_out_file_train=Form("protons_mva2_train.csv");
	//TString str_out_file_test=Form("protons_mva2_test.csv");
	//
	TString str_out_file=Form("%s.csv",str_filename.Data());
	TString str_out_file_train=Form("%s_train.csv",str_filename.Data());
	TString str_out_file_test=Form("%s_test.csv",str_filename.Data());

	ofstream fout;
	ofstream fout_train;
	ofstream fout_test;
  	fout.open(str_out_file.Data());
  	fout_train.open(str_out_file_train.Data());
  	fout_test.open(str_out_file_test.Data());
	
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

	Float_t st_x;
	Float_t st_y;
	Float_t st_z;
	Float_t end_x;
	Float_t end_y;
	Float_t end_z;

	Float_t pbdt;
	Int_t nd;

	Float_t keffbeam;
	Float_t keffhy;
	Float_t kend_bb;


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

	tr->SetBranchAddress("st_x",&st_x);
	tr->SetBranchAddress("st_y",&st_y);
	tr->SetBranchAddress("st_z",&st_z);
	tr->SetBranchAddress("end_x",&end_x);
	tr->SetBranchAddress("end_y",&end_y);
	tr->SetBranchAddress("end_z",&end_z);
	tr->SetBranchAddress("pbdt",&pbdt);
	tr->SetBranchAddress("nd",&nd);
	tr->SetBranchAddress("keffbeam",&keffbeam);
	tr->SetBranchAddress("keffhy",&keffhy);
	tr->SetBranchAddress("kend_bb",&kend_bb);


	//save header to the csv file --------------------------------------------------------//
	//TString txt_out="train,tag,target,ntrklen,trklen,PID,B,costheta,mediandedx,endpointdedx,calo,avcalo";
	TString txt_out="train,tag,target,ntrklen,trklen,PID,B,costheta,mediandedx,endpointdedx,calo,avcalo,st_x,st_y,st_z,end_x,end_y,end_z,pbdt,nd,keffbeam,keffhy,kend_bb, dkeffbeam_bb, dkeffbeam_calo, dkeffhy_bb, dkeffhy_calo, r_keffhy_keffbeam";
  	//fout<<"train,tag,target,ntrklen,trklen,PID,B,costheta,mediandedx,endpointdedx,calo,avcalo\n";
  	//fout_train<<"tag,target,ntrklen,trklen,PID,B,costheta,mediandedx,endpointdedx,calo,avcalo\n";
  	//fout_test<<"tag,target,ntrklen,trklen,PID,B,costheta,mediandedx,endpointdedx,calo,avcalo\n";

	fout<<txt_out.Data()<<"\n";
	fout_train<<txt_out.Data()<<"\n";
	fout_test<<txt_out.Data()<<"\n";



	//loop over all events ------------------------------------//
	Long64_t nentries = tr->GetEntries();
	std::cout<<"nentries: "<<nentries<<std::endl;

	for(int i=0; i<tr->GetEntries(); i++){
	//for(int i=0; i<20; i++){
	   tr->GetEntry(i);
           if (i%1000==0) std::cout<<i<<"/"<<nentries<<std::endl;
	   //if (i<10) cout<<"trklen: "<<trklen<<endl;

           bool isTestSample = true;
           if (i%2 == 0) {
           	isTestSample = false; //Divide MC sample by 2 parts: test+ufold
                        train=0;
           }
	   int TARGET=0;
	   if (tag==1) TARGET=1;
	   		

	   //fout<<train<<","<<tag<<","<<TARGET<<","<<ntrklen<<","<<trklen<<","<<PID<<","<<B<<","<<costheta<<","<<mediandedx<<","<<endpointdedx<<","<<calo<<","<<avcalo<<"\n";	
	   //if (!isTestSample) fout_train<<tag<<","<<TARGET<<","<<ntrklen<<","<<trklen<<","<<PID<<","<<B<<","<<costheta<<","<<mediandedx<<","<<endpointdedx<<","<<calo<<","<<avcalo<<"\n";	
	   //if (isTestSample) fout_test<<tag<<","<<TARGET<<","<<ntrklen<<","<<trklen<<","<<PID<<","<<B<<","<<costheta<<","<<mediandedx<<","<<endpointdedx<<","<<calo<<","<<avcalo<<"\n";	

	   fout<<train<<","<<tag<<","<<TARGET<<","<<ntrklen<<","<<trklen<<","<<PID<<","<<B<<","<<costheta<<","<<mediandedx<<","<<endpointdedx<<","<<calo<<","<<avcalo<<","<<st_x<<","<<st_y<<","<<st_z<<","<<end_x<<","<<end_y<<","<<end_z<<","<<pbdt<<","<<nd<<","<<keffbeam<<","<<keffhy<<","<<kend_bb<<","<<keffbeam-kend_bb<<","<<keffbeam-calo<<","<<keffhy-kend_bb<<","<<keffhy-calo<<","<<keffhy/keffbeam<<"\n";	
	   if (!isTestSample) fout_train<<tag<<","<<TARGET<<","<<ntrklen<<","<<trklen<<","<<PID<<","<<B<<","<<costheta<<","<<mediandedx<<","<<endpointdedx<<","<<calo<<","<<avcalo<<","<<st_x<<","<<st_y<<","<<st_z<<","<<end_x<<","<<end_y<<","<<end_z<<","<<pbdt<<","<<nd<<","<<keffbeam<<","<<keffhy<<","<<kend_bb<<","<<keffbeam-kend_bb<<","<<keffbeam-calo<<","<<keffhy-kend_bb<<","<<keffhy-calo<<","<<keffhy/keffbeam<<"\n";
	   if (isTestSample) fout_test<<tag<<","<<TARGET<<","<<ntrklen<<","<<trklen<<","<<PID<<","<<B<<","<<costheta<<","<<mediandedx<<","<<endpointdedx<<","<<calo<<","<<avcalo<<","<<st_x<<","<<st_y<<","<<st_z<<","<<end_x<<","<<end_y<<","<<end_z<<","<<pbdt<<","<<nd<<","<<keffbeam<<","<<keffhy<<","<<kend_bb<<","<<keffbeam-kend_bb<<","<<keffbeam-calo<<","<<keffhy-kend_bb<<","<<keffhy-calo<<","<<keffhy/keffbeam<<"\n";
	}
  	fout.close();
	fout_train.close();
	fout_test.close();
	   

}
