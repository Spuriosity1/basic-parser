#ifndef basic_parse_hh
#define basic_parse_hh


#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <fstream>
#include <cctype>
#include <iostream>
#include <filesystem>
#include <algorithm>


/**
 * A barebones parameter parser for scientific simulations.
 * 
 * This is intended as a drop-in replacement for long strings of command line
 * arguments that lose their meaning and become confusing.
 * 
 * The main tools are `declare` and `save_file`:
 * 
 * Name your temperature with
 * 
 *      basic_parser p;
 * 
 *      double T; 
 *      p.declare("Temperature (K)", T);
 * 
 *      then 
 *      p.from_file('input.txt')
 * 
 * where 'input.txt' reads
 * 
 *      # this is a comment
 *      Temperature (K): 420
 * 
 * 
 * This basically implents a subset of the TOML standard, but with only top-level directories.
 * 
 * Only top level structure is supported, for more complicated inputs
 * you should just use a library for TOML/YAML/JSON files.
 * 
 */


namespace basic_parser {


typedef uint32_t bp_uint_t;
typedef int32_t bp_int_t;
typedef double bp_float_t;



// CONCEPTS for deducing return type
template <typename T, typename... U>
concept IsAnyOf = (std::same_as<T, U> || ...);

template <typename T>
concept Int = std::is_signed<T>::value && std::is_integral<T>::value;

template <typename T>
concept UInt = !std::is_signed<T>::value && std::is_integral<T>::value && !std::is_same<T, bool>::value;

template <typename T>
concept Float = std::is_floating_point<T>::value;

template <typename T>
concept Bool = std::is_same<T, bool>::value;

template <typename T>
concept String = std::is_same<T, std::string>::value;

template <typename T>
concept Path = std::is_same<T, std::filesystem::path>::value;

template <typename T>
concept valid_input_type = Int<T> || UInt<T> || Bool<T> || String<T> || Path<T> || Float<T>;

/**
 * @brief Returns current date and time formatted as YYYY-MM-DD.HH:mm:ss
 * @return current date/time, format is YYYY-MM-DD.HH:mm:ss
 */
const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

/**
 * @brief Like a python type, just a label
 * 
 */
enum class paramtype {
    None, Int, UInt, Float, String, Bool, ValidPath
};

std::string to_string(paramtype p){
    switch (p)
    {
    case paramtype::Int:
        return "Integer";
    case paramtype::UInt:
        return "Unsigned Integer";
    case paramtype::Float:
        return "Float";
    case paramtype::String:
        return "String";
    case paramtype::Bool:
        return "Boolean";
    case paramtype::ValidPath:
        return "Path";    
    default:
        return "NoneType";
    }
}


/**
 * @brief Basic parameter parser
 * This works in the procedural paradigm - it attaches references from outside to string handles,
 * which must be named at compile time. The only state stored by this class is the set of flags 
 * indicating whether or not basic_parser have been initialised.
 * 
 */
class Parser {
    
    friend std::ostream& operator<<(std::ostream& os, Parser p);
    
    friend std::istream& operator>>(std::istream& is, Parser p);
public:
    Parser(unsigned major_version, unsigned minor_version, char comment_char='#', char output_delimiter='%') :
        major_version(major_version), minor_version(minor_version), comment_char(comment_char), output_delimiter(output_delimiter) {};

    template<valid_input_type T>
    void set_value(const char* handle, T value){
        *(datamap<T>().at(handle)) = value;
        this->initialised[handle] = true;
    }


    /**
     * @brief Outputs current parser state into human readable text
     * 
     * @param os std::ostream to dump data into
     * @param delimiter What should go between the data handle and the data, e.g. `handle::data`
     */
    void into_stream(std::ostream& os, const char* delimiter="=");

    /**
     * @brief Parses data from stream
     * 
     * @param is std::istream to read data from
     * @param delimiter What should go between the data handle and the data, e.g. `handle::data`
     */
    void from_stream(std::istream& is, const char* delimiter="=");

    /**
     * @brief Reads an input file, setting all of the known variables using it
     * 
     * @param fname Filename (C string)
     * @param delimiter Delimiter (C string)
     */
    void from_file(const char* fname, const char* delimiter="=");
    void from_file(const std::string& fname, const char* delimiter="="){
        from_file(fname.c_str(), delimiter);
    }

    /**
     * @brief Interprets command line options in the form --handle=55
     * 
     * @param argc The end of the argv array
     * @param argv it's argv
     * @param start=1 Expects argv[start] to be "--handle1".
     * 
     */
    void from_argv(int argc, const char** argv, int start=1);

    /**
     * @brief Reads an input file, setting all of the known variables using it
     * 
     * @param fname Filename (C string)
     * @param delimiter Delimiter (C string)
     */
    void into_file(const char* fname, const char* delimiter="=");

    void into_file(const std::string& fname, const char* delimiter="="){
        into_file(fname.c_str(), delimiter);
    }


    bool all_initialised(std::initializer_list<const char*>) const;
    bool any_initialised(std::initializer_list<const char*>) const;

    /**
     * @brief Throws an exception if parameters have not been set
     * 
     */
    void assert_all_initialised() const;

    // /**
    //  * @brief Only passes silently if one of the following conditions are met:
    //  *   1. All parameters in v1 are initialised AND no parameters of v2 are initialised.
    //  *   2. All parameters in v2 are initialised AND no parameters of v1 are initialised.
    //  *   All parameters MUST have been declared to make this valid.
    //  * @param v1 
    //  * @param v2 
    //  */
    // void assert_exclusive(const std::vector<std::string>& v1, const std::vector<std::string>& v2);

    /**
     * @brief Associates the C string 'handle' with the variable x.
     * 
     * @param handle The C string to search for in the infile.
     * @param x The varaible that the parser should store the result in (passed by reference)
     */
    template<valid_input_type T>
    const void declare(const std::string& handle, T* x){
        assert_unique(handle);
        this->datamap<T>()[handle] = x;
        index[handle] = paramtype::Int;
        initialised[handle]=false;
    }

    /**
     * @brief Like declare(handle, x), with a default parameter.
     * 
     * @param handle The C string to search for in the infile.
     * @param x The variable that the parser should store the result in (passed by reference).
     * @param default_x The default value of x.
     * 
     * @see void declare(const std::string& handle, bp_int_t& x)
     */
    template<valid_input_type T>
    const void declare_optional(const std::string& handle, T* x, bp_int_t default_x){
        declare(handle, x);
        *x = default_x;
        initialised[handle]=true;
    }


    /**
     * @brief Lists all parameter handles.
     * 
     * @return A nicely formatted string of all names expected from the input file.
     */
    std::string cparam_names(const char* delimiter="=") const{
        std::stringstream s;
        for (auto& [handle, v] : ints) {
            s <<  handle << delimiter << "\t[ int ] ";
            if (initialised.at(handle)) s<<*v;
            s << "\n" ;
        }
        for (auto& [handle, v] : uints) {
            s <<  handle << delimiter << "\t[uint ] ";
            if (initialised.at(handle)) s<<*v;
            s << "\n" ;
        }
        for (auto& [handle, v] : floats) {
            s <<  handle << delimiter << "\t[float] ";
            if (initialised.at(handle)) s<<*v;
            s << "\n" ;
        }
        for (auto& [handle, v] : bools) {
            s <<  handle << delimiter << "\t[bool ] ";
            if (initialised.at(handle)) s<<*v;
            s << "\n" ;
        }
        for (auto& [handle, v] : strings) {
            s <<  handle << delimiter << "\t[ str ] ";
            if (initialised.at(handle)) s<<*v;
            s << "\n" ;
        }
        for (auto& [handle, v] : paths) {
            s <<  handle << delimiter << "\t[path ] ";
            if (initialised.at(handle)) s<<*v;
            s << "\n" ;
        }
        return s.str();
    }


private:
    std::map<const std::string,  bp_int_t*     > ints;
    std::map<const std::string,  bp_uint_t*    > uints;
    std::map<const std::string,  bp_float_t*   > floats;
    std::map<const std::string,  std::string* > strings; 
    std::map<const std::string,  bool*        > bools;
    std::map<const std::string,  std::filesystem::path* > paths; 

    const unsigned major_version;
    const unsigned minor_version;
    const char comment_char;
    const char output_delimiter;
    const char* executable = "";

    bool set_value(const std::string& handle, const std::string& value);

    void assert_unique(const std::string& handle){
        // Make sure we aren't doubling up
        if (index.find(handle)!= index.end()){
            std::cerr<<"Duplicate Declaration found!"<<std::endl;
            throw std::range_error("Duplicate entry");
        }
    }

    std::map<const std::string, bool> initialised; /// Keeps track of whether the variable has been set or not

    std::map<const std::string, paramtype> index; /// Stores the order in which handles were passed for consistency

    
    template <Bool T>
    auto datamap(){return bools;}

    template <Int T>
    auto datamap(){return ints;}

    template <UInt T>
    auto datamap(){return uints;}

    template <Float T>
    auto datamap(){return floats;}

    template <String T>
    auto datamap(){return strings;}

    template <Path T>
    auto datamap(){return paths;}

    
};

std::string strip(const std::string& str)
{
    std::string s(str);
    s.erase(0,s.find_first_not_of(" \t\n\r\f\v"));
    s.erase(s.find_last_not_of(" \t\n\r\f\v") + 1);
    return s;
}


void Parser::from_stream(std::istream& is, const char* delimiter){
    size_t lineno = 0;
    for (std::string line; std::getline(is, line); ){
        lineno++;
        // remove all whitespace from the line
        line = strip(line);
        // Ignore comments and blank lines
        if ( line[0] == comment_char || line.size() == 0) continue;
        
        // Look for delimiters
        size_t idx = line.find(delimiter);

        // If we could not find it, raise an exception
        if (idx == std::string::npos){
            fprintf(stderr, "Invalid line found:\n%03lu | %s", lineno, line.c_str()); fflush(stderr);
            throw std::runtime_error("Invalid file format");
        }

        std::string start = strip(line.substr(0,idx));
        std::string end = strip(line.substr(idx+1));

        try
        {
            if (!set_value(start, end)){
                fprintf(stderr, "Invalid specification on line %03lu:\n '%s'\n", lineno, line.c_str()); 
                fprintf(stderr, "Expected basic_parser format:\n%s\n",cparam_names().c_str());
                fflush(stderr);
                throw std::runtime_error("Invalid specification");
            }
        }
        catch(const std::exception& e)
        {   
            std::cerr<< "Bad input file: " << e.what() << "\n"; 
            std::cerr<< "Expected basic_parser format:\n"<< cparam_names() << std::endl;
            throw e;
            
        }
       
    }
}

bool stobool(const std::string& s){
    std::string ss(s);
    for (char& c : ss){
        c = std::tolower(c);
    }
    if (s == "true"){
        return true;
    } else if (s=="false") {
        return false;
    } else {
        throw std::runtime_error("No valid conversion from string to bool");
    }
}


bool Parser::set_value(const std::string& handle, const std::string& value){
     // see if we recognise the index
        switch (index[handle])
        {
        case paramtype::Int:
            *ints[handle] = stoll(value);
            break;
        case paramtype::UInt:
            *uints[handle] = stoull(value);
            break;
        case paramtype::Float:
            if ( (value.front() == '"' && value.back() == '"') 
                || (value.front() == '\'' && value.back() == '\'') )
            {
                    *floats[handle] = stod(value.substr(1,value.length()-2));
            } else {
                *floats[handle] = stod(value);
            }
            break;
        case paramtype::Bool:
            *bools[handle] = stobool(value);
            break;
        case paramtype::String:
            if ( (value.front() == '"' && value.back() == '"') 
                || (value.front() == '\'' && value.back() == '\'') )
            {
                *strings[handle] = std::string_view(value).substr(1,value.length()-2);
            } else {
                *strings[handle] = strip(value);
            }
            break;
        case paramtype::ValidPath:
            if ( (value.front() == '"' && value.back() == '"') 
                || (value.front() == '\'' && value.back() == '\'') )
            {
                *paths[handle] = std::string_view(value).substr(1,value.length()-2);
            } else {
                *paths[handle] = strip(value);
            }
            if (!std::filesystem::exists(*paths.at(handle))){
                throw std::runtime_error("No such file or directory: " + (*paths[handle]).string());
            }
            break;
        default:
            return false;
        }
        initialised[handle] = true;
        return true;
}



void Parser::into_stream(std::ostream& os, const char* delimiter){
    
    os << comment_char << " --- OUTPUT FILE --- \n";
    os << comment_char << " " << executable << " v"<<major_version << "."<<minor_version<<"\n";

    for (auto& [handle, type] : index){
        switch (type)
        {
        case paramtype::Int:
            os << handle <<delimiter<< *ints[handle]<<"\n";
            break;
        case paramtype::UInt:
            os << handle <<delimiter<< *uints[handle]<<"\n";
            break;
        case paramtype::Float:
            os << handle <<delimiter<< *floats[handle]<<"\n";
            break;
        case paramtype::String:
            os << handle <<delimiter<<'"'<< *strings[handle]<<"\"\n";
            break;
        case paramtype::Bool:
            os << handle <<delimiter<< (*bools[handle]? "\"true\"" : "\"false\"")<<"\n";
            break;
        default:
            os << handle <<delimiter<< "unknown! \n";
            break;
        }
    }
}


std::ostream& operator<<(std::ostream& os, Parser p){
    p.into_stream(os);
    return os;
}


std::istream& operator>>(std::istream& is, Parser p){
    p.from_stream(is);
    return is;
}


void Parser::into_file(const char* fname, const char* delimiter){
    std::ofstream ofs(fname);
    if (!ofs.is_open()) {
        fprintf(stderr, "Error opening file %s\n", fname);
        throw std::runtime_error("Cannot open file");
    }
    try{
        into_stream(ofs, delimiter);
        ofs.close();
    } catch  (const char* e) {
        fprintf(stderr, "Error writing file: %s\n",e);
        throw std::runtime_error("Cannot write to file");
        ofs.close();
    }
}
    

void Parser::from_file(const char* fname, const char* delimiter){
    std::ifstream ifs(fname);
    if (!ifs.is_open()) {
        fprintf(stderr, "Error opening file!\n");
    }
    try{
        from_stream(ifs, delimiter);
        ifs.close();
    } catch  (const char* e) {
        fprintf(stderr, "Error reading file: %s\n",e);
        ifs.close();
        throw e;
    }
}


/**
 * @brief Parses command line arguments, starting from argv[start]
 * 
 * @param argc Number of nonzero command line arguments (including the ones skipped by start)
 * @param argv Argument vector
 * @param start First index to check
 * @return std::string Command line overrides in format %arg1=val1%arg2=val2 etc.
 */
void
Parser::from_argv(int argc, const char** argv, int start)
{
    this->executable = argv[0];

    for (int i=start; i<argc; i++)
    {
        std::string s(argv[i]);
        if (s[0] != '-' || s[1] != '-')
        {
            throw std::runtime_error("Bad keyword argument: handles must be prefixed with '--'");
        }
        s.erase(0,2);

        // Look for delimiters
        size_t idx = s.find('=');
        // If we could not find it, raise an exception
        if (idx == std::string::npos){
            std::cerr << "Invalid kwarg: "<< s <<std::endl;
            throw std::runtime_error("Invalid kwarg");
        }
        
        std::string start = strip(s.substr(0,idx));
        std::string end = strip(s.substr(idx+1));

        try
        {
            bool success = set_value(start, end);
            if (!success)
            {
                std::cerr<< "Invalid kwarg at position" << i<<": "<<s<<"\n";
                throw std::runtime_error("Invalid kwarg");
            }
        }
        catch(const std::exception& e)
        {   
            std::cerr<< "Bad kwarg at position "<<i<<": " << e.what() << "\n"; 
            throw std::runtime_error("Invalid kwarg");
        }
    }
}



void 
Parser::assert_all_initialised() const {
    bool giveup=false;
    for (const auto& [k, x] : initialised){
        if (x == false){
            fprintf(stderr, "Uninitialied %s: %s\n",to_string(index.at(k)).c_str(),k.c_str());
            giveup=true;
        }
    }
    if (giveup){
        fprintf(stderr, "Expected format:\n%s",cparam_names().c_str());
        throw std::runtime_error("Mandatory variables missing from infile");
    }
}



bool Parser::all_initialised(std::initializer_list<const char*> handles) const 
{
    for (auto& h : handles){
        if (!this->initialised.at(std::string(h))) return false;
    }
    return true;
}


bool Parser::any_initialised(std::initializer_list<const char*> handles) const 
{
    for (auto& h : handles){
        if (!this->initialised.at(std::string(h))) return false;
    }
    return true;
}








}; // end of namespace basic_parser


#endif