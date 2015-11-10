# Gap Analysis for Determining K-mean Clustering
## Myeong Lee
### University of Maryland College Park (iSchool)

The code is to determine K number in K-mean clustering using gap analysis method.
The original code was developed by DataScienceLab (https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/).

Since the original code was targeting (1) only 2-tuple vectors; and (2) was not maintaining vector IDs so to track the data.
My modified implementation tackled these two issues. 

There are two sets of functions to include vector IDs: with and without a prefix "new_". If a function begins with "new_", that function is for maintaining IDs. If not, the fuction is not maintaining IDs. 
The functions are work well with n-dimensional vectors. 

Feel free to use/modify the code. 
Any questions? (deeperlee@gmail.com)
