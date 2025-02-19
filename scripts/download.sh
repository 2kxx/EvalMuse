git lfs install
git -c http.extraHeader="User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:95.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3" clone https://hf-mirror.com/datasets/DY-Evalab/EvalMuse
cd EvalMuse
cat images.zip.part-* > images.zip
unzip -d ../ images.zip
mv *.json ../datasets
