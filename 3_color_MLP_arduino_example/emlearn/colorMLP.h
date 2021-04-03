#include "eml_net.h"
static const float colorMLP_layer0_weights[15] = { -0.227409f, -0.039437f, 0.017645f, 0.015578f, 0.082042f, -0.005759f, -0.022325f, 0.012583f, 0.021661f, -0.257275f, 0.164368f, -0.018416f, 0.039534f, 0.031181f, 0.152049f };
static const float colorMLP_layer0_biases[5] = { 0.872096f, -0.428815f, -0.119577f, -0.418876f, 1.561883f };
static const float colorMLP_layer1_weights[30] = { -4.716386f, 3.169954f, -4.356012f, -4.189064f, 3.036406f, 2.771722f, -0.038060f, -0.024014f, 0.000142f, 0.006355f, 0.028073f, 0.018317f, -0.252473f, -1.861514f, 1.356843f, 1.166160f, -0.488273f, -0.107324f, -0.234296f, -1.143462f, 0.818715f, 1.166673f, 0.086330f, -0.309856f, 4.116200f, 3.862642f, -3.792697f, -4.310450f, -3.938827f, -3.990081f };
static const float colorMLP_layer1_biases[6] = { -0.196522f, -1.346803f, 0.672995f, 0.611209f, -0.269148f, -0.262125f };
static float colorMLP_buf1[6];
static float colorMLP_buf2[6];
static const EmlNetLayer colorMLP_layers[2] = { 
{ 5, 3, colorMLP_layer0_weights, colorMLP_layer0_biases, EmlNetActivationLogistic }, 
{ 6, 5, colorMLP_layer1_weights, colorMLP_layer1_biases, EmlNetActivationSoftmax } };
static EmlNet colorMLP = { 2, colorMLP_layers, colorMLP_buf1, colorMLP_buf2, 6 };
