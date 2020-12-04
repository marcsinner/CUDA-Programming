#include "aux.h"
#include "jpeglib.h"
#include <boost/program_options.hpp>
#include <iostream>
#include <unordered_map>

namespace po = boost::program_options;


ProgramSettins getInputFromCmd(int argc, char *argv[]) {
    ProgramSettins settings{};

    po::options_description desc("Allowed options");

    real centerX{}, centerY{}, domainWidth{}, domainHeight{};
    std::string modeStr{};
  // clang-format off
    desc.add_options()("help,h", "produce help message")
        ("center_x,x", po::value<real>(&centerX)->default_value(-0.75), "domain center along x")
        ("center_y,y", po::value<real>(&centerY)->default_value(-0.25), "domain center along y")
        ("width,W", po::value<real>(&domainWidth)->default_value(3.5), "domain width")
        ("height,H", po::value<real>(&domainHeight)->default_value(4.0), "domain height")
        ("it", po::value<uint32_t>(&settings.maxIterations)->default_value(1000), "max. num. iterations")
        ("resX", po::value<size_t>(&settings.numPixelsX)->default_value(6400), "num. pixels along x direction")
        ("resY", po::value<size_t>(&settings.numPixelsY)->default_value(4800), "num. pixels along y direction")
        ("quality,q", po::value<unsigned>(&settings.imageQuality)->default_value(100), "image quality")
        ("output,o", po::value<std::string>(&settings.outputFileName)->default_value("./set.jpeg"), "output file")
        ("mode,m", po::value<std::string>(&modeStr)->default_value("serial"), "supported compute mode: serial, threading, gpu");
  // clang-format on
    po::variables_map cmdMap;
    po::store(po::parse_command_line(argc, argv, desc), cmdMap);
    po::notify(cmdMap);

    if (cmdMap.count("help")) {
        std::cout << desc << "\n";
        exit(EXIT_SUCCESS);
    }

    settings.domain.minX = centerX - 0.5 * domainWidth;
    settings.domain.maxX = centerX + 0.5 * domainWidth;

    settings.domain.minY = centerY - 0.5 * domainHeight;
    settings.domain.maxY = centerY + 0.5 * domainHeight;

    std::unordered_map<std::string, MODE> modeMap = {
        {"serial", MODE::Serial}, {"threading", MODE::Threading}, {"gpu", MODE::GPU}};

    if (modeMap.find(modeStr) != modeMap.end()) {
        settings.mode = modeMap[modeStr];
    } else {
        throw std::runtime_error("ERROR: incorrect mode provided. Allowed: serial, threading, gpu");
    }

    return settings;
}


std::string modeToString(MODE mode) {
    switch (mode) {
        case MODE::Serial:
            return std::string("serial");
        case MODE::Threading:
            return std::string("threading");
        case MODE::GPU:
            return std::string("gpu");
        default:
            throw std::runtime_error("ERROR: unknwon compute mode type");
    }
}


void checkInputData(const ProgramSettins &settings) {
    if ((settings.numPixelsX < 0) || settings.numPixelsY < 0) {
        throw std::runtime_error(
            "ERROR: negative input parameters for the output image resolution");
    }

    real width = settings.domain.maxX - settings.domain.minX;
    real height = settings.domain.maxY - settings.domain.minY;
    if ((width < 0) || (height < 0)) {
        throw std::runtime_error("ERROR: negative input parameters for the domain");
    }

    if (!((settings.imageQuality >= 10) || (settings.imageQuality <= 100))) {
        throw std::runtime_error("ERROR: image quality should be between 10 and 100");
    }
}


std::ostream &operator<<(std::ostream &stream, const ProgramSettins &data) {
    std::cout << "image resolution along x: " << data.numPixelsX << '\n'
              << "image resolution along y: " << data.numPixelsY << '\n'
              << "min x: " << data.domain.minX << "; max x: " << data.domain.maxX << '\n'
              << "min y: " << data.domain.minY << "; max y: " << data.domain.maxY << '\n'
              << "max num. iterations: " << data.maxIterations << '\n'
              << "image quality: " << data.imageQuality << '\n'
              << "compute mode: " << modeToString(data.mode) << '\n'
              << "size of the floating point type, bytes: " << sizeof(real) << '\n'
              << "output file name: " << data.outputFileName << std::endl;
    return stream;
}


void writeData(const std::vector<unsigned char> &outputField, const ProgramSettins &settings) {

    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;
    FILE *outfile;

    // set up the error handler first, in case the initialization step fails.
    cinfo.err = jpeg_std_error(&jerr);

    // initialize the JPEG compression object.
    jpeg_create_compress(&cinfo);

    if ((outfile = fopen(settings.outputFileName.c_str(), "wb")) == NULL) {
        std::cout << "cannot open: " << settings.outputFileName << std::endl;
    }
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = settings.numPixelsX;
    cinfo.image_height = settings.numPixelsY;
    cinfo.input_components = 1;

    cinfo.in_color_space = JCS_GRAYSCALE;
    jpeg_set_defaults(&cinfo);

    jpeg_set_quality(&cinfo, settings.imageQuality, TRUE);

    // start compression
    jpeg_start_compress(&cinfo, TRUE);

    JSAMPLE *imageBuffer = const_cast<unsigned char *>(outputField.data());
    JSAMPROW rowPointer[1];
    while (cinfo.next_scanline < cinfo.image_height) {
        // jpeg_write_scanlines expects an array of pointers to scanlines.
        // Here the array is only one element long, but you could pass
        // more than one scanline at a time if that's more convenient.
        rowPointer[0] = &imageBuffer[cinfo.next_scanline * settings.numPixelsX];
        (void)jpeg_write_scanlines(&cinfo, rowPointer, 1);
    }

    // Finish compression
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
}