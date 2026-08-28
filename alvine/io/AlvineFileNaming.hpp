    std::string sanitizeFileToken(std::string token) const {
        for (char& c : token) {
            if (!std::isalnum(static_cast<unsigned char>(c))) {
                c = '_';
            }
        }
        return token;
    }

    std::string dtFileToken() const {
        std::ostringstream os;
        os << std::setprecision(12) << dt_file_label_m;
        std::string token = os.str();

        if (token.find('.') != std::string::npos) {
            while (!token.empty() && token.back() == '0') {
                token.pop_back();
            }
            if (!token.empty() && token.back() == '.') {
                token.pop_back();
            }
        }

        return sanitizeFileToken(token);
    }

    std::string diagnosticFileName(const std::string& baseName) const {
        return sanitizeFileToken(method_m) + "_dt_" + dtFileToken() + "_" + baseName;
    }
