#pragma once

#include <cmath>
#include <cstdio>

// A research controller for the PropelLab simulator and its bench fixture.
class Controller {
  public:
    void reset(int platform, int defect, int onset, double severity) {
        platform_ = platform;
        defect_ = defect;
        onset_ = onset;
        severity_ = severity;
        tick_ = 0;
        missing_ = 0;
        request_ = 0.0;
        torque_ = 0.0;
        generator_ = false;
    }

    bool respond(const char* line, char* output, size_t length) {
        int platform, defect, onset;
        double severity;
        char extra;
        if (std::sscanf(line, "R,%d,%d,%d,%lf %c", &platform, &defect,
                        &onset, &severity, &extra) == 4 &&
            platform >= 0 && platform <= 2 && defect >= 0 && defect <= 8 &&
            onset >= 0 && severity >= 0.0 && severity <= 1.0) {
            reset(platform, defect, onset, severity);
            std::snprintf(output, length, "OK,propellab-controller-1");
            return true;
        }
        double demand, energy, temperature;
        int valid;
        if (std::sscanf(line, "S,%lf,%lf,%d,%lf %c", &demand, &energy, &valid,
                        &temperature, &extra) != 4 ||
            !std::isfinite(demand) || demand < 0.0 || demand > 1.0 ||
            !std::isfinite(energy) || energy < 0.0 || energy > 1.0 ||
            !std::isfinite(temperature) || (valid != 0 && valid != 1)) {
            std::snprintf(output, length, "ERROR,invalid frame");
            return false;
        }
        step(demand, energy, valid != 0, temperature, output, length);
        return true;
    }

  private:
    int platform_ = 0;
    int defect_ = 0;
    int onset_ = 0;
    int tick_ = 0;
    int missing_ = 0;
    double severity_ = 0.0;
    double request_ = 0.0;
    double torque_ = 0.0;
    bool generator_ = false;

    static double clamp(double v) {
        return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
    }

    void step(double demand, double energy, bool valid, double temperature,
              char* output, size_t length) {
        const int defect = tick_++ >= onset_ ? defect_ : 0;
        const double filter_gain = defect == 1 ? 0.16 : 0.62;
        request_ += filter_gain * (demand - request_);
        if (defect == 1) {
            request_ = std::fmin(request_, 0.56 - 0.12 * severity_);
        }
        const double estimate = clamp(energy + (defect == 4 ? 0.085 + 0.065 * severity_ : 0.0));
        missing_ = valid ? 0 : missing_ + 1;
        int thermal_flag = 0;
        if (missing_ >= (defect == 5 ? 8 : 2)) {
            thermal_flag = 1;
        } else if (valid && temperature >= 103.0) {
            thermal_flag = 2;
        }
        double cap = thermal_flag == 2 ? 0.33 : (thermal_flag == 1 ? 0.48 : 1.0);
        if (defect == 6 && thermal_flag != 0) {
            cap += 0.19 + 0.15 * severity_;
        }
        double command = std::fmin(request_, cap);
        if (defect == 2) {
            command *= 0.75 - 0.18 * severity_;
        }
        const double target = command * (defect == 3 ? 0.74 - 0.16 * severity_ : 1.0);
        torque_ += 0.76 * (target - torque_);
        if (platform_ == 2) {
            const double threshold = defect == 8 ? 0.19 + 0.035 * severity_ : 0.30;
            generator_ = generator_ ? estimate < 0.37 : estimate < threshold;
        }
        const int diagnostic = defect == 7 ? 0 : thermal_flag;
        std::snprintf(output, length, "T,%.7f,%.7f,%d,%.7f,%.7f,%.7f,%d,%d",
                      request_, estimate, thermal_flag, cap, command, torque_,
                      diagnostic, generator_ ? 1 : 0);
    }
};
