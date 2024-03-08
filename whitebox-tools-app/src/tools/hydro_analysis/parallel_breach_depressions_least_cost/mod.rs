#[allow(dead_code)]
mod algorithms;
mod search;
#[allow(dead_code)]
mod structures;
use self::algorithms::parallel_breach_depressions_least_cost;
use crate::tools::*;
use ordered_float::OrderedFloat;
use std::io::{Error, ErrorKind};
use std::path::PathBuf;

pub struct ParallelBreachDepressionsLeastCost {
    name: String,
    description: String,
    toolbox: String,
    parameters: Vec<ToolParameter>,
    example_usage: String,
}

impl ParallelBreachDepressionsLeastCost {
    pub fn new() -> ParallelBreachDepressionsLeastCost {
        // public constructor
        let name = "ParallelBreachDepressionsLeastCost".to_string();
        let toolbox = "Hydrological Analysis".to_string();
        let description =
            "Breaches the depressions in a DEM using a least-cost pathway method with parallelized pathway search.".to_string();

        let mut parameters = vec![];
        parameters.push(ToolParameter {
            name: "Input DEM File".to_owned(),
            flags: vec!["-i".to_owned(), "--dem".to_owned()],
            description: "Input raster DEM file.".to_owned(),
            parameter_type: ParameterType::ExistingFile(ParameterFileType::Raster),
            default_value: None,
            optional: false,
        });

        parameters.push(ToolParameter {
            name: "Output File".to_owned(),
            flags: vec!["-o".to_owned(), "--output".to_owned()],
            description: "Output raster file.".to_owned(),
            parameter_type: ParameterType::NewFile(ParameterFileType::Raster),
            default_value: None,
            optional: false,
        });

        parameters.push(ToolParameter {
            name: "Maximum Search Distance (cells)".to_owned(),
            flags: vec!["--dist".to_owned()],
            description: "Maximum search distance for breach paths in cells.".to_owned(),
            parameter_type: ParameterType::Integer,
            default_value: None,
            optional: false,
        });

        parameters.push(ToolParameter {
            name: "Maximum Breach Cost (z units)".to_owned(),
            flags: vec!["--max_cost".to_owned()],
            description: "Optional maximum breach cost (default is Inf).".to_owned(),
            parameter_type: ParameterType::Float,
            default_value: None,
            optional: true,
        });

        parameters.push(ToolParameter {
            name: "Minimize breach distances?".to_owned(),
            flags: vec!["--min_dist".to_owned()],
            description: "Optional flag indicating whether to minimize breach distances."
                .to_owned(),
            parameter_type: ParameterType::Boolean,
            default_value: Some("true".to_string()),
            optional: true,
        });

        parameters.push(ToolParameter {
            name: "Flat increment value (z units)".to_owned(),
            flags: vec!["--flat_increment".to_owned()],
            description: "Optional elevation increment applied to flat areas.".to_owned(),
            parameter_type: ParameterType::Float,
            default_value: None,
            optional: true,
        });

        parameters.push(ToolParameter {
            name: "Fill unbreached depressions?".to_owned(),
            flags: vec!["--fill".to_owned()],
            description:
                "Optional flag indicating whether to fill any remaining unbreached depressions."
                    .to_owned(),
            parameter_type: ParameterType::Boolean,
            default_value: Some("true".to_string()),
            optional: true,
        });

        let sep: String = path::MAIN_SEPARATOR.to_string();
        let e = format!("{}", env::current_exe().unwrap().display());
        let mut parent = env::current_exe().unwrap();
        parent.pop();
        let p = format!("{}", parent.display());
        let mut short_exe = e
            .replace(&p, "")
            .replace(".exe", "")
            .replace(".", "")
            .replace(&sep, "");
        if e.contains(".exe") {
            short_exe += ".exe";
        }
        let usage = format!(
            ">>.*{0} -r={1} -v --wd=\"*path*to*data*\" --dem=DEM.tif -o=output.tif --dist=1000 --max_cost=100.0 --min_dist",
            short_exe, name
        )
        .replace("*", &sep);

        ParallelBreachDepressionsLeastCost {
            name: name,
            description: description,
            toolbox: toolbox,
            parameters: parameters,
            example_usage: usage,
        }
    }
}

impl WhiteboxTool for ParallelBreachDepressionsLeastCost {
    fn get_source_file(&self) -> String {
        String::from(file!())
    }

    fn get_tool_name(&self) -> String {
        self.name.clone()
    }

    fn get_tool_description(&self) -> String {
        self.description.clone()
    }

    fn get_tool_parameters(&self) -> String {
        match serde_json::to_string(&self.parameters) {
            Ok(json_str) => return format!("{{\"parameters\":{}}}", json_str),
            Err(err) => return format!("{:?}", err),
        }
    }

    fn get_example_usage(&self) -> String {
        self.example_usage.clone()
    }

    fn get_toolbox(&self) -> String {
        self.toolbox.clone()
    }

    fn run<'a>(
        &self,
        args: Vec<String>,
        working_directory: &'a str,
        verbose: bool,
    ) -> Result<(), Error> {
        let parsed_args = parse_args(args)?;
        let validated_args = validate_args(parsed_args, working_directory)?;
        println!("Validated arguments:");
        dbg!(validated_args.clone());
        println!("working directory: {}", working_directory);
        println!("verbose: {}", verbose);

        println!("Running ParallelBreachDepressionsLeastCost...");
        match parallel_breach_depressions_least_cost(validated_args) {
            Ok(_) => Ok(()),
            Err(error) => Err(Error::new(ErrorKind::Other, error)),
        }
    }
}

#[derive(Debug)]
struct ParsedArgs {
    input_file: String,
    output_file: String,
    max_cost: f64,
    max_dist: isize,
    flat_increment: f64,
    fill_deps: bool,
    minimize_dist: bool,
}

impl Default for ParsedArgs {
    fn default() -> Self {
        ParsedArgs {
            input_file: "".to_string(),
            output_file: "".to_string(),
            max_dist: 20isize,
            max_cost: f64::INFINITY,
            flat_increment: 1.0 * 10.0_f64.powi(-10),
            fill_deps: false,
            minimize_dist: false,
        }
    }
}

fn parse_args(args: Vec<String>) -> Result<ParsedArgs, Error> {
    // Split key value args into individual strings
    // e.g. ["--max_dist=20", "--fill"] => ["--max_dist", "20", "--fill"]
    let args: Vec<&str> = args.iter().flat_map(|s| s.split("=")).collect();

    let mut parsed_args = ParsedArgs::default();
    let mut args_iterator = args.into_iter();

    while let Some(arg) = args_iterator.next() {
        let missing_value_msg = format!("Expected value after {}, found none", arg);
        match arg {
            "-i" | "--input" | "--dem" => {
                let next_value = args_iterator.next();
                match next_value {
                    Some(next_value) if !next_value.starts_with("-") => {
                        parsed_args.input_file = next_value.to_owned()
                    }
                    _ => return Err(Error::new(ErrorKind::InvalidInput, missing_value_msg)),
                }
            }
            "-o" | "--output" => {
                let next_value = args_iterator.next();
                match next_value {
                    Some(next_value) if !next_value.starts_with("-") => {
                        parsed_args.output_file = next_value.to_owned()
                    }
                    _ => return Err(Error::new(ErrorKind::InvalidInput, missing_value_msg)),
                }
            }
            "--dist" => {
                let next_value = args_iterator.next();
                match next_value {
                    Some(next_value) if !next_value.starts_with("-") => {
                        let parsed_int = next_value.parse::<isize>().expect(
                            format!("Expected an integer for max_dist, found '{}'", next_value)
                                .as_str(),
                        );
                        parsed_args.max_dist = parsed_int;
                    }
                    _ => return Err(Error::new(ErrorKind::InvalidInput, missing_value_msg)),
                }
            }
            "--max_cost" => {
                let next_value = args_iterator.next();
                match next_value {
                    Some(next_value) => {
                        let parsed_float = next_value.parse::<f64>().expect(
                            format!("Expected an float for max_dist, found '{}'", next_value)
                                .as_str(),
                        );
                        parsed_args.max_cost = parsed_float;
                    }
                    _ => return Err(Error::new(ErrorKind::InvalidInput, missing_value_msg)),
                }
            }
            "--flat_increment" => {
                let next_value = args_iterator.next();
                match next_value {
                    Some(next_value) => {
                        let parsed_float = next_value.parse::<f64>().expect(
                            format!("Expected an float for max_dist, found '{}'", next_value)
                                .as_str(),
                        );
                        parsed_args.flat_increment = parsed_float;
                    }
                    _ => return Err(Error::new(ErrorKind::InvalidInput, missing_value_msg)),
                }
            }
            "--fill" => parsed_args.fill_deps = true,
            "--min_dist" => parsed_args.minimize_dist = true,
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!("Found unexpected argument: '{}'", arg),
                ))
            }
        }
    }
    Ok(parsed_args)
}

#[derive(Debug, Clone)]
struct ValidatedArgs {
    input_file: String,
    output_file: String,
    max_cost: OrderedFloat<f64>,
    max_dist: usize,
    flat_increment: OrderedFloat<f64>,
    fill_deps: bool,
    minimize_dist: bool,
    num_threads: usize,
}

fn validate_args(parsed_args: ParsedArgs, working_directory: &str) -> Result<ValidatedArgs, Error> {
    let input_file =
        if parsed_args.input_file.contains("/") || parsed_args.input_file.contains("\\") {
            PathBuf::from(parsed_args.input_file)
        } else {
            PathBuf::from(working_directory).join(PathBuf::from(parsed_args.input_file))
        };

    if !input_file.exists() {
        panic!(
            "Input file does not exist at {}",
            input_file.to_str().unwrap()
        )
    }

    let output_file =
        if parsed_args.output_file.contains("/") || parsed_args.output_file.contains("\\") {
            PathBuf::from(parsed_args.output_file)
        } else {
            PathBuf::from(working_directory).join(PathBuf::from(parsed_args.output_file))
        };

    match output_file.parent() {
        Some(dir) => {
            if !dir.exists() {
                panic!("Output file directory does not exist")
            }
        }
        None => panic!("Output file directory does not exist"),
    }
    if !(parsed_args.max_cost > 0f64) {
        panic!("max_cost must be greater than zero")
    }
    if !(parsed_args.max_dist > 0isize) {
        panic!("max_dist must be greater than zero")
    }
    if !(parsed_args.flat_increment > 0f64) {
        panic!("flat_increment must be greater than zero")
    }

    // Calculate the number of available threads
    let max_procs = whitebox_common::configs::get_configs()?.max_procs;
    let num_threads = match max_procs {
        isize::MIN..=-1 => num_cpus::get(),
        0..=1 => panic!("num_threads must be at least 2"),
        _ => std::cmp::min(max_procs as usize, num_cpus::get()),
    };

    let validated_args = ValidatedArgs {
        input_file: String::from(input_file.to_str().unwrap()),
        output_file: String::from(output_file.to_str().unwrap()),
        max_cost: OrderedFloat(parsed_args.max_cost),
        max_dist: parsed_args.max_dist as usize,
        flat_increment: OrderedFloat(parsed_args.flat_increment),
        fill_deps: parsed_args.fill_deps,
        minimize_dist: parsed_args.minimize_dist,
        num_threads: num_threads,
    };
    Ok(validated_args)
}
