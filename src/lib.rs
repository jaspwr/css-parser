use std::f32;

pub type StyleSheet = Vec<Rule>;

#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    pub selector: Selector,
    pub properties: Vec<Property>,
}

impl Rule {
    pub fn matches_classes(&self, classes: &Vec<String>) -> bool {
        self.selector.matches_classes(classes)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Property {
    pub name: String,
    pub value: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Selector {
    None,
    Tag(String),
    Class(String),
    ID(String),
    Pseudo(String),
    Descendant(Box<Selector>, Box<Selector>),
    Child(Box<Selector>, Box<Selector>),
    NextSibling(Box<Selector>, Box<Selector>),
    All,
}

impl Selector {
    pub fn matches_classes(&self, classes: &Vec<String>) -> bool {
        match self {
            Selector::Class(s) => classes.contains(s),
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CSSParseError {
    pub location: usize,
    pub message: String,
}

#[derive(PartialEq, Debug, Copy, Clone)]
enum ParsingContext {
    Selector,
    PropertyName,
    AwaitingColon,
    PropertyValue,
}

fn assert_state(
    state: ParsingContext,
    expected: ParsingContext,
    token: &str,
    pos: usize,
) -> Result<(), CSSParseError> {
    if state != expected {
        return Err(CSSParseError {
            location: pos,
            message: format!("Unexpected {:?}", token),
        });
    }
    Ok(())
}

fn parse_selector(s: &[Token], pos: usize) -> Result<Selector, CSSParseError> {
    assert!(!s.is_empty());

    if s.len() == 1 {
        if let Token::Identifier(s) = s[0] {
            if s == "*" {
                return Ok(Selector::All);
            }

            if s.starts_with("#") {
                if s.len() < 2 {
                    return Err(CSSParseError {
                        location: pos,
                        message: "Empty ID name.".to_string(),
                    });
                }
                return Ok(Selector::ID(s[1..].to_string()));
            }
            if s.starts_with(".") {
                if s.len() < 2 {
                    return Err(CSSParseError {
                        location: pos,
                        message: format!("Empty class name: {:?}", s),
                    });
                }
                return Ok(Selector::Class(s[1..].to_string()));
            }
            if s.starts_with(":") {
                if s.len() < 2 {
                    return Err(CSSParseError {
                        location: pos,
                        message: "Empty Pseudo name.".to_string(),
                    });
                }
                return Ok(Selector::Pseudo(s[1..].to_string()));
            }

            assert!(s.len() > 0);

            return Ok(Selector::Tag(s.to_string()));
        }
    }

    let a = parse_selector(&s[0..1], pos)?;

    match s[1] {
        Token::Identifier(oper) => {
            if oper == ">" {
                let b = parse_selector(&s[2..], pos)?;
                return Ok(Selector::Child(Box::new(a), Box::new(b)));
            }
            if oper == "+" {
                let b = parse_selector(&s[2..], pos)?;
                return Ok(Selector::NextSibling(Box::new(a), Box::new(b)));
            }
        }
        _ => {}
    }

    let b = parse_selector(&s[1..], pos)?;
    return Ok(Selector::Descendant(Box::new(a), Box::new(b)));
}

fn parse(input: &str, mut state: ParsingContext) -> Result<StyleSheet, CSSParseError> {
    let mut pos = 0;

    let mut ss: StyleSheet = vec![];
    let init_state = state;

    let mut current_style = Rule {
        selector: Selector::None,
        properties: vec![],
    };

    let mut current_property = Property {
        name: String::new(),
        value: vec![],
    };

    macro_rules! add_prop {
        () => {
            if !current_property.name.is_empty() && !current_property.value.is_empty() {
                current_style.properties.push(current_property.clone());
            }
            current_property.name.clear();
            current_property.value.clear();
        };
    }

    let mut selector_tokens = vec![];

    while let Some(token) = next_token(input, &mut pos, state)? {
        if state == ParsingContext::AwaitingColon {
            if token != Token::Colon {
                return Err(CSSParseError {
                    location: pos,
                    message: format!("Expected ':', found {:?}", token),
                });
            }
            state = ParsingContext::PropertyValue;
            continue;
        }

        match token {
            Token::OpenBrace => {
                assert_state(state, ParsingContext::Selector, "{", pos)?;
                state = ParsingContext::PropertyName;

                current_style.selector = parse_selector(&selector_tokens, pos)?;
                selector_tokens.clear();
            }
            Token::CloseBrace => {
                assert_state(state, ParsingContext::PropertyName, "}.", pos)?;
                ss.push(current_style.clone());
                current_style.properties.clear();
                state = ParsingContext::Selector;
            }
            Token::Colon => {
                assert_state(state, ParsingContext::Selector, ":", pos)?;
                selector_tokens.push(token);
            }
            Token::Semicolon => {
                assert_state(state, ParsingContext::PropertyValue, ";", pos)?;
                add_prop!();
                state = ParsingContext::PropertyName;
            }
            Token::Identifier(s) => match state {
                ParsingContext::Selector => {
                    selector_tokens.push(Token::Identifier(s));
                }
                ParsingContext::PropertyName => {
                    current_property.name = s.to_string();
                    state = ParsingContext::AwaitingColon;
                }
                ParsingContext::PropertyValue => {
                    // println!("Adding value: {} -> {:?}", s, parse_value(&s));
                    current_property.value.push(parse_value(&s));
                }
                ParsingContext::AwaitingColon => {
                    unreachable!();
                }
            },
            Token::String(s) => {
                assert_state(state, ParsingContext::PropertyValue, "string", pos)?;
                current_property.value.push(Value::String(s.to_string()));
            }
        }
    }

    // HACK
    if init_state != ParsingContext::Selector {
        add_prop!();
        ss.push(current_style.clone());
    }

    if init_state == ParsingContext::Selector && state != ParsingContext::Selector {
        return Err(CSSParseError {
            location: pos,
            message: "Unbalanced curly braces".to_string(),
        });
    }

    Ok(ss)
}

pub fn parse_full(input: &str) -> Result<StyleSheet, CSSParseError> {
    parse(input, ParsingContext::Selector)
}

pub fn parse_properties(input: &str) -> Result<Vec<Property>, CSSParseError> {
    // HACK
    parse(input, ParsingContext::PropertyName).map(|s| {
        assert_eq!(s.len(), 1);
        s[0].properties.clone()
    })
}

#[derive(PartialEq, Debug, Clone)]
enum Token<'a> {
    Identifier(&'a str),
    String(&'a str),
    Colon,
    Semicolon,
    OpenBrace,
    CloseBrace,
}

fn skip_whitespace_and_commnets(input: &str, pos: &mut usize) {
    let mut in_comment = false;

    if input[*pos..].starts_with("/*") {
        in_comment = true;
        *pos += 2;
    }

    while *pos < input.len() && (in_comment || input.chars().nth(*pos).unwrap().is_whitespace()) {
        if in_comment {
            if input[*pos..].starts_with("*/") {
                in_comment = false;
                *pos += 2;
                continue;
            }
        } else {
            if input[*pos..].starts_with("/*") {
                in_comment = true;
                *pos += 2;
                continue;
            }
        }

        *pos += 1;
    }
}

fn next_token<'src>(
    input: &'src str,
    pos: &mut usize,
    ctx: ParsingContext,
) -> Result<Option<Token<'src>>, CSSParseError> {
    skip_whitespace_and_commnets(input, pos);
    if *pos >= input.len() {
        return Ok(None);
    }

    let start = *pos;

    let c = input.chars().nth(*pos).unwrap();
    if c == '{' {
        *pos += 1;
        return Ok(Some(Token::OpenBrace));
    }
    if c == '}' {
        *pos += 1;
        return Ok(Some(Token::CloseBrace));
    }
    if c == ':' && ctx != ParsingContext::Selector {
        *pos += 1;
        return Ok(Some(Token::Colon));
    }
    if c == ';' {
        *pos += 1;
        return Ok(Some(Token::Semicolon));
    }
    if c == '"' {
        let unclosed_opener_pos = Some(*pos);
        *pos += 1;
        let mut escaped = false;
        while *pos < input.len() {
            let c = input.chars().nth(*pos).unwrap();
            if !escaped && c == '\\' {
                escaped = true;
            } else if escaped {
                escaped = false;
            } else if c == '"' {
                *pos += 1;
                return Ok(Some(Token::String(&input[start..*pos])));
            }
            *pos += 1;
        }
        return Err(CSSParseError {
            location: unclosed_opener_pos.unwrap_or(*pos),
            message: "Unterminated string".to_string(),
        });
    }

    let mut seen_non_colon = false;
    while *pos < input.len() {
        let c = input.chars().nth(*pos).unwrap();

        if c != ':' {
            seen_non_colon = true;
        }

        if c == '(' {
            let unclosed_opener_pos = Some(*pos);
            let mut bracket_depth = 0;
            *pos += 1;
            while *pos < input.len() {
                let c = input.chars().nth(*pos).unwrap();
                if c == '(' {
                    bracket_depth += 1;
                } else if bracket_depth == 0 && c == ')' {
                    *pos += 1;
                    return Ok(Some(Token::Identifier(&input[start..*pos])));
                } else if c == ')' {
                    bracket_depth -= 1;
                }
                *pos += 1;
            }
            return Err(CSSParseError {
                location: unclosed_opener_pos.unwrap_or(*pos),
                message: "Unclosed backet".to_string(),
            });
        }

        if !id_char(c) && !(ctx == ParsingContext::Selector && c == ':' && !seen_non_colon) {
            break;
        }

        *pos += 1;
    }

    if *pos > start {
        return Ok(Some(Token::Identifier(&input[start..*pos])));
    }

    Ok(None)
}

fn id_char(c: char) -> bool {
    !c.is_whitespace() && c != '"' && c != ':' && c != ';' && c != '{' && c != '}'
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    String(String),
    /// Radians
    Angle(f32),
    Number(f32),
    Length(Length),
    Colour(Colour),
    LinearGradient {
        angle: f32,
        points: Vec<GradientPoint>,
    },
    Variable(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Length {
    Px(f32),
    Percentage(f32),
    Auto,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GradientPoint {
    pub pos: f32,
    pub col: Value,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Colour {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Colour {
    pub fn as_hex(&self) -> String {
        format!(
            "{:02x}{:02x}{:02x}{:02x}",
            (self.r * 255.) as u8,
            (self.g * 255.) as u8,
            (self.b * 255.) as u8,
            (self.a * 255.) as u8
        )
    }
}

fn parse_value(value: &str) -> Value {
    let value = value.trim();

    if let Ok(v) = value.parse::<f32>() {
        return Value::Number(v);
    }

    if value.starts_with("#") {
        if let Some(c) = parse_colour(value) {
            return Value::Colour(c);
        }
    }

    if value.ends_with("rad") {
        if let Ok(v) = value.trim_end_matches("rad").parse::<f32>() {
            return Value::Angle(v);
        }
    }

    if value.ends_with("deg") {
        if let Ok(v) = value.trim_end_matches("deg").parse::<f32>() {
            return Value::Angle(v.to_radians());
        }
    }

    if value.ends_with("px") {
        if let Ok(v) = value.trim_end_matches("px").parse::<f32>() {
            return Value::Length(Length::Px(v));
        }
    }

    if value.ends_with("%") {
        if let Ok(v) = value.trim_end_matches("%").parse::<f32>() {
            return Value::Length(Length::Percentage(v));
        }
    }

    if value == "auto" {
        return Value::Length(Length::Auto);
    }

    if value == "transparent" {
        return Value::Colour(Colour {
            r: 0.,
            g: 0.,
            b: 0.,
            a: 0.,
        });
    }

    if value.starts_with("var") {
        if let Some(args) = css_function_args(value) {
            if args.len() == 1 {
                return Value::Variable(args[0].to_string());
            }
        }
    }

    if value.starts_with("linear-gradient") || value.starts_with("gradient") {
        if let Some(args) = css_function_args(value) {
            let (angle, points) = parse_gradient(args.iter().map(|s| s.as_str()).collect());

            return Value::LinearGradient { angle, points };
        }
    }

    Value::String(value.to_string())
}

fn parse_gradient(args: Vec<&str>) -> (f32, Vec<GradientPoint>) {
    let mut angle = 0.;
    let mut points = vec![];

    for values in args
        .into_iter()
        .map(|v| v.split(' ').map(parse_value).collect::<Vec<_>>())
    {
        if values.is_empty() {
            continue;
        }

        if let Value::Angle(ang) = values.first().unwrap() {
            angle = *ang;

            if angle.is_nan() {
                angle = 20.;
            }
        } else {
            let mut point = GradientPoint {
                col: values.first().unwrap().clone(),
                pos: f32::NAN,
            };

            if let Some(Value::Number(n)) = values.get(1) {
                point.pos = *n;
            }

            if let Some(Value::Length(Length::Percentage(p))) = values.get(1) {
                point.pos = p / 100.
            }

            points.push(point);
        }
    }

    let points_count = points.len();
    for (i, point) in points.iter_mut().enumerate() {
        if point.pos.is_nan() {
            point.pos = (i as f32) / (points_count as f32 - 1.).max(1.);
        }
    }

    (angle, points)
}

fn css_function_args(func: &str) -> Option<Vec<String>> {
    let mut args = vec![];
    let mut current_arg = String::new();
    let mut bracket_depth = 0;

    for c in func.chars() {
        if c == '(' {
            bracket_depth += 1;
        } else if c == ')' {
            bracket_depth -= 1;
        } else if c == ',' && bracket_depth == 1 {
            if !current_arg.trim().is_empty() {
                args.push(current_arg.trim().to_string());
            }
            current_arg.clear();
            continue;
        }

        if bracket_depth > 0 && !(c == '(' && bracket_depth == 1) {
            current_arg.push(c);
        }
    }
    if !current_arg.trim().is_empty() {
        args.push(current_arg.trim().to_string());
    }

    Some(args)
}

pub fn parse_colour(col: &str) -> Option<Colour> {
    let col = col.trim().trim_start_matches("#");

    if col.len() == 3 {
        let r = &col[0..1];
        let g = &col[1..2];
        let b = &col[2..3];
        return Some(Colour {
            r: u8::from_str_radix(r, 16).ok()? as f32 / 15.,
            g: u8::from_str_radix(g, 16).ok()? as f32 / 15.,
            b: u8::from_str_radix(b, 16).ok()? as f32 / 15.,
            a: 1.,
        });
    }

    if col.len() != 6 && col.len() != 8 {
        return None;
    }

    let r = u8::from_str_radix(&col[0..2], 16).ok()? as f32 / 255.;
    let g = u8::from_str_radix(&col[2..4], 16).ok()? as f32 / 255.;
    let b = u8::from_str_radix(&col[4..6], 16).ok()? as f32 / 255.;

    let a = if col.len() == 8 {
        u8::from_str_radix(&col[6..8], 16).ok()? as f32 / 255.
    } else {
        1.
    };

    Some(Colour { r, g, b, a })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn props() {
        let props = parse_properties("").unwrap();
        assert_eq!(props, vec![]);

        let props = parse_properties("foo: 1rad").unwrap();
        assert_eq!(
            props,
            vec![Property {
                name: "foo".to_string(),
                value: vec![Value::Angle(1.)]
            },]
        );

        let props = parse_properties("foo: 1rad; bar: var(--baz)").unwrap();

        println!("{:#?}", props);

        assert_eq!(
            props,
            vec![
                Property {
                    name: "foo".to_string(),
                    value: vec![Value::Angle(1.)]
                },
                Property {
                    name: "bar".to_string(),
                    value: vec![Value::Variable("--baz".to_string())]
                }
            ]
        )
    }

    #[test]
    fn full() {
        let stylesheet = parse_full(
            "
           .class {
                background-color: #fff;
            }

            #bleh {
                font-size: var(--foo); 
            }

            #bleh:hover {
                opacity: 0.2;
            }

            ::root {
                --foo: 2px;
            }
        ",
        )
        .unwrap();

        println!("{:#?}", stylesheet);

        assert_eq!(
            stylesheet,
            vec![
                Rule {
                    selector: Selector::Class("class".to_string()),
                    properties: vec![Property {
                        name: "background-color".to_string(),
                        value: vec![Value::Colour(Colour {
                            r: 1.,
                            g: 1.,
                            b: 1.,
                            a: 1.
                        })]
                    },]
                },
                Rule {
                    selector: Selector::ID("bleh".to_string()),
                    properties: vec![Property {
                        name: "font-size".to_string(),
                        value: vec![Value::Variable("--foo".to_string())]
                    },]
                },
                Rule {
                    selector: Selector::Descendant(
                        Box::new(Selector::ID("bleh".to_string())),
                        Box::new(Selector::Pseudo("hover".to_string()))
                    ),
                    properties: vec![Property {
                        name: "opacity".to_string(),
                        value: vec![Value::Number(0.2)]
                    },]
                },
                Rule {
                    selector: Selector::Pseudo(":root".to_string()),
                    properties: vec![Property {
                        name: "--foo".to_string(),
                        value: vec![Value::Length(Length::Px(2.))]
                    },]
                },
            ]
        );
    }
}
