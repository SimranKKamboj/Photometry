procedure tvload(common,starfile,tnofile)

        string common {"", prompt=" Common portion of filename "}
        string starfile {"", prompt=" Name of coordinates file "}
        string tnofile {"", prompt=" End of TNO coordinates file name "}

begin 
        int    nimg , firstimg, currframe
        string infile,rootword, coostar, cootno, cootnopf

        print(" ")
        print(" ") 
        print(" Welcome to MY LOADing program ")
        print(" ") 
        rootword = common
        print(" first image to load : ") 
        scan(firstimg)
        print(" # of images to load : ") 
        scan(nimg)
        coostar = starfile 
        cootno = tnofile
        # starfile = "relative.star"  # Assuming common.star is in the current directory

        print(" ") 
        print(" Loading files: ")
        print(" ") 
        for (i=firstimg; i<=nimg+firstimg-1; i+=1)
        {
                currframe = i - firstimg + 1
                infile = rootword // i
                print(" Currently loading ", infile, " into frame ", currframe)

                if(currframe < 17) 
                    display(infile, currframe, fill-)
                else
                    display(infile, 16, fill-)

                cootnopf = infile//cootno
                # Mark coordinates on the current frame
                tvmark(currframe,coostar,label-,number-,color=204)
                tvmark(currframe,cootnopf,label-,number-,color=205)
        }

end
