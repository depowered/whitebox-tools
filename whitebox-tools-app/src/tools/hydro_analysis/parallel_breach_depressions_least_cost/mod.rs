#[allow(dead_code)]
mod algorithms;
#[allow(dead_code)]
mod structures;
use crate::tools::*;
use std::io::Error;

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
        println!("Running ParallelBreachDepressionsLeastCost...");
        for arg in args {
            println!("arg: {}", arg);
        }
        println!("working directory: {}", working_directory);
        println!("verbose: {}", verbose);
        Ok(())
    }
}
