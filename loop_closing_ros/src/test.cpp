// citation:https://github.com/nicolov/simple_slam_loop_closure load_filenames
#include "LoopClosingTool.hpp"
using namespace std;

vector<std::string> load_filenames(const std::string& dir,
                                        const bool skip_even = true) {
  std::vector<std::string> filenames;

  auto index_filename = dir + "/fileindex.csv";

  std::cout << "Opening index from " << index_filename << "\n";

  std::ifstream f;
  f.open(index_filename);

  if (!f) {
    throw std::runtime_error("Failed to open index file");
  }

  while (!f.eof()) {
    std::string l;
    getline(f, l);

    if (!l.empty()) {
      std::stringstream ss;
      ss << l;
      std::string filename;
      ss >> filename;
      filenames.push_back(filename);
    }

    // Discard even-numbered images (for stereo datasets)
    if (skip_even) {
      std::getline(f, l);
    }
  }

  return filenames;
}
void Eigen_to_File(const Eigen::MatrixXd* matrix_,string filename){
  std::ofstream out;
  out.open(filename.c_str());
  int r = matrix_->rows();
  int c = matrix_->cols();
  for (int i = 0; i < r; i ++){
    for (int j = 0; j < c; j ++){
      out << (*matrix_)(i,j) << " ";
    }
    out << endl;
  }
  out.close();
}
int main(int argc,char *argv[]) {
    
    // Step1： read in image 
    // Step2: Dbow2 config
     // create CvoLoopClosing instance
    
    //read in vovabulary file
    string dataPath = "../data";
    string imagePath = "../../loop_closing_data";
    DBoW3::Vocabulary voc(dataPath + "/orbvoc.dbow3");
    DBoW3::Database db(voc, false, 0);

    //read in image index
    auto fileindex = load_filenames(dataPath,false);
    int totalSequence = fileindex.size();
   
    // //config window
    // namedWindow( "Display window", cv::WINDOW_AUTOSIZE ); 
    vector<string> filenames;
    //process image 1 by 1
    //intialize loopclosing tool
    Eigen::MatrixXd loopClosureMatch = Eigen::MatrixXd::Zero(totalSequence,totalSequence);
    LoopClosingTool loopDetector(&db,&loopClosureMatch); 
    for (int i = 0; i < totalSequence;i++){
        string filename = imagePath + "/" + fileindex[i]+".png";
        //IC(filename);
        //read in new image
        cv::Mat currentImg = cv::imread(filename);
        loopDetector.assignNewFrame(currentImg);
        loopDetector.create_feature();
        loopDetector.detect_loop();
        //test descriptor mode
        //loopDetector.detect_loop(currentImg);


        cv::imshow("dispaly",currentImg);
        cv::waitKey(10);
        filenames.push_back(filename);
    }
    // Loop closing tool to file
    Eigen_to_File(&loopClosureMatch,dataPath+"/result.txt");

    
}