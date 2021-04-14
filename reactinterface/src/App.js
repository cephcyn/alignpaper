import React from 'react';

import logo from './logo.svg';
import './App.css';

class ShiftButton extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      could_shift: false
    };
    this.shiftButton = this.shiftButton.bind(this);
  }

  componentDidMount() {
    this.setState({ could_shift: false });
    // set whether this shift button is enabled or not
    fetch("/api/alignop/canshift?"+new URLSearchParams({
        alignment: JSON.stringify(this.props.data),
        row: this.props.rownum,
        col: this.props.colnum,
        shift_dist: this.props.direction,
      }))
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.setState({ could_shift: data.is_legal });
      });
  }

  componentDidUpdate(prevProps) {
    if (this.props.data !== prevProps.data) {
      this.componentDidMount();
    }
  }

  shiftButton(e) {
    e.preventDefault();
    // console.log("Shift button clicked!");
    // console.log(e);
    fetch("/api/alignop/shift?"+new URLSearchParams({
        alignment: JSON.stringify(this.props.data),
        row: this.props.rownum,
        col: this.props.colnum,
        shift_dist: this.props.direction,
      }))
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.props.onAlignmentChange(data);
      });
  }

  render() {
    // console.log("rerendering ShiftButton =========");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    let text;
    if (this.props.direction === -1) {
      text = "<"; // &lt;
    } else {
      text = ">"; // &gt;
    }

    return (
      <button
        disabled={!this.state.could_shift}
        onClick={this.shiftButton}>
          {text}
      </button>)
  }
}

class AlignmentTable extends React.Component {
  render() {
    // console.log("rerendering AlignmentTable =========");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    const rows = this.props.data.map(
      (row) => {
        const cols = row.txt.map((cell, index) => {
          return (
            <td key={index}>
              <ShiftButton
                data={this.props.data}
                rownum={row.id}
                colnum={index}
                direction={-1}
                onAlignmentChange={this.props.onAlignmentChange}
              />
              {cell.join(' ')}
              <ShiftButton
                data={this.props.data}
                rownum={row.id}
                colnum={index}
                direction={1}
                onAlignmentChange={this.props.onAlignmentChange}
              />
            </td>
          );
        });
        return (
          <tr key={row.id}>
            <td className="align-id">{row.id}</td>
            {cols}
          </tr>
        );
      }
    );

    return (
      <table>
        <tbody>
          {rows}
        </tbody>
      </table>
    );
  }
}

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      alignment: [],
      inputvalue: "",
      loading: false,
    };
    this.handleTextChange = this.handleTextChange.bind(this);
    this.handleAlignmentChange = this.handleAlignmentChange.bind(this);
    this.alignRawText = this.alignRawText.bind(this);
    this.alignmentScore = this.alignmentScore.bind(this);
    this.alignmentSearch = this.alignmentSearch.bind(this);
    this.buttonDoesNothing = this.buttonDoesNothing.bind(this);
  }

  handleTextChange(e) {
    this.setState({inputvalue: e.target.value});
  }

  handleAlignmentChange(e) {
    this.setState({alignment: e.alignment});
  }

  alignRawText(e) {
    e.preventDefault();
    console.log("Raw text align button clicked!");
    console.log(e);
    console.log("value=");
    console.log(this.state.inputvalue);
    this.setState({ loading: true });
    fetch("/api/textalign?"+new URLSearchParams({
      input: this.state.inputvalue,
      // id: "3",
    }))
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.setState({ loading: false });
        this.setState(data);
      });
  }

  alignmentScore(e) {
    e.preventDefault();
    console.log("alignment score button clicked!");
    console.log(e);
    this.setState({ loading: true });
    fetch("/api/alignscore?"+new URLSearchParams({
      alignment: JSON.stringify(this.state.alignment),
    }))
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.setState({ loading: false });
        this.setState(data);
      });
  }

  alignmentSearch(e) {
    e.preventDefault();
    console.log("alignment search button clicked!");
    console.log(e);
    this.setState({ loading: true });
    fetch("/api/alignsearch?"+new URLSearchParams({
      alignment: JSON.stringify(this.state.alignment),
    }))
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.setState({ loading: false });
        this.setState(data);
      });
  }

  buttonDoesNothing(e) {
    e.preventDefault();
    console.log("nothing button clicked!");
    console.log(e);
  }

  render() {
    console.log("rerendering App =========", new Date());
    console.log("props:", this.props);
    console.log("state:", this.state);

    // only render waiting spinner if we are currently waiting on the api
    let spinner;
    if (this.state.loading) {
      spinner = <p>Loading...</p>
    } else {
      spinner = <br/>
    }

    // only render alignment if there's content
    let aligntable;
    if (this.state.alignment.length > 0) {
      aligntable = <AlignmentTable
        data={this.state.alignment}
        onAlignmentChange={this.handleAlignmentChange}
      />
    } else {
      aligntable = <br />
    }

    return (
      <div className="App">
        <textarea
          value={this.state.inputvalue}
          onChange={this.handleTextChange}
          className="raw-input"
        />
        <br />
        <button onClick={this.alignRawText}>Align Texts</button>
        <button onClick={this.alignmentScore}>Alignment Score</button>
        <button onClick={this.alignmentSearch}>Alignment Search</button>
        <button onClick={this.buttonDoesNothing}>This Button Does Nothing</button>
        <hr />
        {aligntable}
        {spinner}
        <hr />
        <p>alignment_score is...</p>
        <p>{this.state.alignment_score ? this.state.alignment_score.toString() : 'Undefined'}</p>
        <hr />
        <p>alignment_rawtext is...</p>
        <p>{this.state.alignment_rawtext ? this.state.alignment_rawtext.toString() : 'Undefined'}</p>
        <hr />
        <img src={logo} className="App-logo" alt="logo" />
      </div>
    );
  }
}

export default App;
