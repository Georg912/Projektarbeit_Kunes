version="2.2"


cd Projectarbeit_*/
rm -r 0* Figures/ Modules/ Images_For_Notebooks/
cd ..
mv Projectarbeit_Georg_Hufnagl*/ Projectarbeit_Georg_Hufnagl_v$version/
cp -r 0* Figures/ Images_For_Notebooks/ Modules/ Projectarbeit_Georg_Hufnagl*/
rm Projectarbeit_Georg_Hufnagl*zip
zip -r Projectarbeit_Georg_Hufnagl_v$version.zip Projectarbeit_Georg_Hufnagl*
