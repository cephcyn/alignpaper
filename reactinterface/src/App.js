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
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.props.data),
        row: this.props.rownum,
        col: this.props.colnum,
        shift_dist: this.props.direction,
      })
    };
    fetch("/api/alignop/canshift", requestOptions)
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
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.props.data),
        row: this.props.rownum,
        col: this.props.colnum,
        shift_dist: this.props.direction,
      })
    };
    fetch("/api/alignop/shift", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.props.onAlignmentChange(data);
      });
  }

  render() {
    // console.log("rerendering ShiftButton ..............");
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
        className="tight"
        disabled={!this.state.could_shift}
        onClick={this.shiftButton}>
          {text}
      </button>)
  }
}

class InsertButton extends React.Component {
  constructor(props) {
    super(props);
    this.insertButton = this.insertButton.bind(this);
  }

  insertButton(e) {
    e.preventDefault();
    console.log("Insert button clicked!");
    console.log(e);
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.props.data),
        col: this.props.colnum,
        insertafter: true,
      })
    };
    fetch("/api/alignop/insertcol", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.props.onAlignmentChange(data);
      });
  }

  render() {
    // console.log("rerendering InsertButton ...........");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    return (
      <button
        className="tight"
        onClick={this.insertButton}>
          +
      </button>)
  }
}

class DeleteButton extends React.Component {
  constructor(props) {
    super(props);
    this.deleteButton = this.deleteButton.bind(this);
  }

  deleteButton(e) {
    e.preventDefault();
    console.log("Delete button clicked!");
    console.log(e);
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.props.data),
        col: this.props.colnum,
      })
    };
    fetch("/api/alignop/deletecol", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.props.onAlignmentChange(data);
      });
  }

  render() {
    // console.log("rerendering DeleteButton .............");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    return (
      <button
        className="tight"
        onClick={this.deleteButton}>
          -
      </button>)
  }
}

class AlignmentTable extends React.Component {
  render() {
    // console.log("rerendering AlignmentTable ..............");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    const rows = this.props.data.map(
      (row) => {
        const cols = row.txt.map((cell, index) => {
          return (
            <td key={index}>
              {cell.join(' ')}
              <br/>
              <ShiftButton
                data={this.props.data}
                rownum={row.id}
                colnum={index}
                direction={-1}
                onAlignmentChange={this.props.onAlignmentChange}
              />
              <ShiftButton
                data={this.props.data}
                rownum={row.id}
                colnum={index}
                direction={1}
                onAlignmentChange={this.props.onAlignmentChange}
              />
              <br/>
              <InsertButton
                data={this.props.data}
                colnum={index}
                onAlignmentChange={this.props.onAlignmentChange}
              />
              <DeleteButton
                data={this.props.data}
                colnum={index}
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
    this.updateAlignmentProgress = this.updateAlignmentProgress.bind(this);
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
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({input: this.state.inputvalue})
    };
    fetch("/api/textalign", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.updateAlignmentProgress(data['location']);
      });
  }

  updateAlignmentProgress(status_url) {
    fetch(status_url)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        console.log('got update from server...');
        console.log(data);
        if (data['state'] !== 'PENDING' && data['state'] !== 'PROGRESS') {
          if ('alignment' in data) {
            this.setState({ parse_constituency: data['parse_constituency'] });
            this.setState({ alignment: data['alignment'] });
            this.setState({ loading: false });
          } else {
            this.setState({ alignment: [] });
            this.setState({ loading: false });
          }
        } else {
          // check back on the progress in 2 seconds...
          setTimeout(() => {
            this.updateAlignmentProgress(status_url);
          }, 2000);
        }
      });
  }

  alignmentScore(e) {
    e.preventDefault();
    console.log("alignment score button clicked!");
    console.log(e);
    this.setState({ loading: true });
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.state.alignment),
      })
    };
    fetch("/api/alignscore", requestOptions)
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
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.state.alignment),
      })
    };
    fetch("/api/alignsearch", requestOptions)
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
    console.log("rerendering App.......", new Date());
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
        <img src={logo} className="App-logo" alt="logo" />
      </div>
    );
  }
}

export default App;
