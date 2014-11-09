#include <shotton/shotton.h>

using namespace std;
namespace lab1231_sun_prj {
namespace shotton {

void train(const string datasets_name, EnergyParam* energy_param) {
  cout << "train(): BEGIN\n";

  if (strcmp(datasets_name.c_str(),"VOC")==0) {
    // From [Shotton, 2009                                                 ]
    (*energy_param)["theta_phi_1"] = 4.5;
    (*energy_param)["theta_phi_2"] = 1.0;  
  } 
  else {
    assert(false && "Unknown dataset!");
  }

  cout << "train(): END\n";
}

Eigen::MatrixXi annotate(const size_t n_label, const string img_dir, const string unary_dir, EnergyParam energy_param) {  

  cout << "annotate(): BEGIN\n";

  //read image file
  cv::Mat img_mat = cv::imread(img_dir, CV_LOAD_IMAGE_COLOR);
  ///read unary
  ProbImage unary_mat;
  unary_mat.decompress( unary_dir.c_str() );

  const size_t n_var = img_mat.rows * img_mat.cols;

  GraphicalModel gm( opengm::SimpleDiscreteSpace<size_t, size_t>(n_var, n_label) );

  //printf("%d\n",n_label);

  set_1st_order(img_mat, unary_mat, n_label, gm);
  set_2nd_order(img_mat, n_label, energy_param, gm);

  Eigen::MatrixXi ann(img_mat.rows, img_mat.cols);
  const string method = "AlphaExpansion";//: "AlphaExpansion", "ICM"
  infer(method, gm, n_var, ann);

  cout << "annotate(): END\n";
  return ann;
}

void infer(const string method, GraphicalModel gm, const size_t n_var, Eigen::MatrixXi& ann) {
  cout << "infer(): BEGIN\n";
  cout << "method= " << method << endl;

  vector<size_t> ann_vec(n_var);
  
  if (method=="AlphaExpansion") {
    typedef 
    opengm::external::MinSTCutKolmogorov<size_t, double> 
    MinStCutType;

    typedef 
    opengm::GraphCut<GraphicalModel, opengm::Minimizer, MinStCutType> 
    MinGraphCut;
    
    typedef 
    opengm::AlphaExpansion<GraphicalModel, MinGraphCut> 
    MinAlphaExpansion;

    MinAlphaExpansion inf_engine(gm);

    cout << "Inferring ..." << endl;
    inf_engine.infer();
    inf_engine.arg(ann_vec);
  }
/*  else if (method=="AlphaExpansionFusion") {
    typedef 
    opengm::external::MinSTCutKolmogorov<size_t, double> 
    MinStCutType;

    typedef 
    opengm::GraphCut<GraphicalModel, opengm::Minimizer, MinStCutType> 
    MinGraphCut;
    
    typedef 
    opengm::AlphaExpansionFusion<GraphicalModel, MinGraphCut> 
    MinAlphaExpansionFusion;

    MinAlphaExpansionFusion inf_engine(gm);
  }*/
  else if (method=="ICM") {
    typedef opengm::ICM<GraphicalModel, opengm::Minimizer> IcmType;
    IcmType::VerboseVisitorType visitor;

    IcmType inf_engine(gm);
    inf_engine.infer(visitor);
  }
  else {
    assert(false && "Unknown inference method");
  }

  //
  size_t idx = 0;
  for (size_t i=0; i<ann.rows(); ++i) {
    for (size_t j=0; j<ann.cols(); ++j) {
      //ok no bug
      ann(i,j) = ann_vec.at(idx);
      ++ idx;
    }
  }

  cout << "infer(): END\n";
}

void set_1st_order(const cv::Mat img_mat, ProbImage unary_mat, const size_t n_label, GraphicalModel& gm) {
  using namespace std;
  assert(unary_mat.width()==img_mat.cols && "err");
  assert(unary_mat.height()==img_mat.rows && "err");

  for (size_t x=0; x<img_mat.cols; ++x) {
    for (size_t y=0; y<img_mat.rows; ++y) {
      // add a function
      const size_t shape[] = {n_label};
      opengm::ExplicitFunction<float> energy(shape, shape+1);

      for(size_t i = 0; i < n_label; i++) 
        energy(i) = -unary_mat(x,y,i);  

      GraphicalModel::FunctionIdentifier fid = gm.addFunction(energy);
      
      // add a factor
      size_t var_idxes[] = {util::flat_idx(x, y, img_mat.cols)};
      gm.addFactor(fid, var_idxes, var_idxes+1);
    }
  }
}

void set_2nd_order(cv::Mat img_mat, const size_t n_label, EnergyParam energy_param, GraphicalModel& gm) {
  // Params needed by the Pott model
  const float equal_pen = 0.0;

  //
  float beta;
  beta = edge_potential::get_beta(img_mat);

  Eigen::MatrixXd theta_phi(2, 1);
  theta_phi << energy_param["theta_phi_1"], 
               energy_param["theta_phi_2"];

  //
  for (size_t x=0; x<img_mat.cols; ++x) {
    for (size_t y=0; y<img_mat.rows; ++y) {
      cv::Point2i p1;   
      p1.x = x; p1.y = y;

      // (x, y) -- (x + 1, y)
      if (x+1 < img_mat.cols) {
        // add a function
        cv::Point2i p2;   
        p2.x = x+1; p2.y = y;

        float unequal_pen;
        unequal_pen = edge_potential::potential(img_mat.at<cv::Vec3b>(p1), img_mat.at<cv::Vec3b>(p2), beta, theta_phi);

        //
        opengm::PottsFunction<float> pott(n_label, n_label, equal_pen, unequal_pen);
        GraphicalModel::FunctionIdentifier fid = gm.addFunction(pott);

        // add a factor
        size_t var_idxes[] = {util::flat_idx(x,y,img_mat.cols), util::flat_idx(x+1,y,img_mat.cols)};
        sort(var_idxes, var_idxes + 2);
        gm.addFactor(fid, var_idxes, var_idxes + 2);
      }

      // (x, y) -- (x, y + 1)
      if (y+1 < img_mat.rows) {
        // add a function
        cv::Point2i p2;   
        p2.x = x; p2.y = y+1;

        float unequal_pen;
        unequal_pen = edge_potential::potential(img_mat.at<cv::Vec3b>(p1), img_mat.at<cv::Vec3b>(p2), beta, theta_phi);

        //
        opengm::PottsFunction<float> pott(n_label, n_label, equal_pen, unequal_pen);
        GraphicalModel::FunctionIdentifier fid = gm.addFunction(pott);

        // add a factor
        size_t var_idxes[] = {util::flat_idx(x,y,img_mat.cols), util::flat_idx(x,y+1,img_mat.cols)};
        sort(var_idxes, var_idxes + 2);
        gm.addFactor(fid, var_idxes, var_idxes + 2);
      }
    }
  }
}

}// namespace shotton
} // namespace lab1231_sun_prj