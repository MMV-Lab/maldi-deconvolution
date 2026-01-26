---

https://www.nature.com/articles/s41592-025-02889-8
 
Do you think we could test it for MALDI data? We could provide
Brain MSI at 5 µm laser spot size and start oversampling at 3,2,1 µm. We also have 
A circular profile and all other things should be the same but only
That we image lipids/metabolites. Do you think this would be possible?

---

Hi Jianxu,
 
and the PSF of our instrument in focus has been characterized here
https://www.sciencedirect.com/science/article/pii/S1387380610001028
 
So Chiahsin will test if we can get similar intensity curves as shown in Ext. Figure 1c
Because this will determine how often we can oversample and how to calibrate. Once
done we will decide if it is feasible with MALDI.

---

In the OriginLab file you will find the signal loss curves and their raw x-y pairs. The two Excel sheets contain the signal intensity (m/z 650-850) from every time point (I guess you don’t need them but just in case). 
There are three folders, 2 of them have the imzML files for each laser pass, the other has 2 oversampled measurements. 

Let me explain the metadata shown in my file name:
1.	“A” stands for attenuator, which is controlling the laser energy. In our instrument, higher value gives lower laser.
2.	“xxum” indicates the pixel size, number of pixel is also listed

And about estimating the subpixel areas, I would suggest to assume our laser spot is a 4-µm circle for the first tries, although the craters are not perfectly round in shape. 

I think these are the information you would need. Let me know if I missed something 😊
