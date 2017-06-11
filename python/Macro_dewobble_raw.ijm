//setBatchMode(true);

directory = "/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinalMask/DATA/raw/";  //getArgument(); //getDirectory("Choose a Directory");
print(directory);

parentdir = File.getParent(directory);
print(parentdir);

filenames = getFileList(directory);
print(filenames.length);

for (fidx=0; fidx<filenames.length; fidx++) {
	
	
	//n("Image Sequence...", "open=directory+filenames[0]");
	open(directory+filenames[fidx]);
	
	newfilepath =substring(filenames[fidx], 0, lengthOf(filenames[fidx]) - 4);
	print(newfilepath);
	File.makeDirectory(parentdir + "/" + newfilepath);
	
	
	run("Deinterleave", "how=2");
	
	//filenames = getFileList(directory);
	nchannels = 2;
	imlist = newArray(nchannels);
	for (c=1; c<=nchannels; c++) {
		selectImage(c);
		imlist[c-1] = getTitle();
	}

	for (f=0; f<imlist.length; f++) {
	
	
		currdir = imlist[f] + "/";
		subsavepath = parentdir + "/" + newfilepath + "/" + currdir;
		File.makeDirectory(subsavepath);
		print(subsavepath);
		
		imname = imlist[f]; //substring(currdir, 0, lengthOf(currdir)-1);
		
		print(imname);
		selectWindow(imname);
	
	    
			for (i=0; i<340; i++) {
				index = i;
				selectWindow(imname);
				
				if (i==339) {
					run("MultiStackReg", "stack_1=imname action_1=[Load Transformation File] file_1=[/nas/volume1/2photon/RESDATA/phase1_block2/alignment2/MSR_avg_transform.txt] stack_2=None action_2=Ignore file_2=[] transformation=[Rigid Body]");
					rename("Substack_" + index + ".tif");
					//saveAs("Tiff", "/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/DATA/" +  "/" +getTitle());
					saveAs("Tiff", subsavepath +  getTitle());
					close();
					
				} else{
					
				
				run("Make Substack...", "delete slices=1-22");
				run("MultiStackReg", "stack_1=[Substack (1-22)] action_1=[Load Transformation File] file_1=[/nas/volume1/2photon/RESDATA/phase1_block2/alignment2/MSR_avg_transform.txt] stack_2=None action_2=Ignore file_2=[] transformation=[Rigid Body]");
				
				//run("Tiff...");
				rename("Substack_" + index + ".tif");
			
				//saveAs("Tiff", "/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/DATA/" + filename + "/" +getTitle());
				saveAs("Tiff", subsavepath + getTitle());
				//run("Save");
				close();
				}
			}
	
	}
}

